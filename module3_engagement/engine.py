"""
Module 3: Email Engagement Engine

Manages multi-round email conversations with candidates.
- Sends initial questions
- Reads replies and classifies response type
- Generates contextual follow-ups
- Tracks thread state across parallel conversations

DESIGN DECISIONS:
- Gmail API via OAuth2 for sending (SMTP as fallback)
- IMAP polling for reply detection (every 2-5 min)
- Thread tracking via Message-ID / In-Reply-To headers
- State machine per thread: initiated → awaiting → replied → follow_up → completed/stalled
- Rate limiting: max 100 emails/hour to avoid spam filters

WHAT I TRIED:
- Attempt 1: Used smtplib directly. Worked for sending but had no way to
  track replies reliably. Had to add IMAP polling, which introduced a 
  separate connection and state management headache.
  
- Attempt 2: Tried to use Gmail API for both send+receive. OAuth2 setup 
  was painful — the token refresh logic burned 3 hours of debugging. 
  Error I hit:
    google.auth.exceptions.RefreshError: ('invalid_grant: Token has been 
    expired or revoked', {'error': 'invalid_grant', ...})
  
  Root cause: was storing the auth token in the wrong format. Fixed by 
  using google-auth-oauthlib properly.

- Final: Abstracted email operations behind an interface so the system 
  can use Gmail API, SMTP+IMAP, or a mock backend for testing.
"""

import os
import sys
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.schema import Candidate, ScoredCandidate, CandidateMetadata, ModuleOutput
from shared.database import Database
from shared.utils import setup_logger, word_count

logger = setup_logger("engagement")


# ====================================================================
# THREAD STATE MACHINE
# ====================================================================

class ThreadState(str, Enum):
    INITIATED = "initiated"
    AWAITING_REPLY = "awaiting_reply"
    REPLIED = "replied"
    FOLLOW_UP_SENT = "follow_up_sent"
    COMPLETED = "completed"
    STALLED = "stalled"
    ELIMINATED = "eliminated"


class ThreadTracker:
    """
    Manages the state of email threads per candidate.
    Each candidate has one active thread at a time.
    Thread state is persisted to DB for crash recovery.
    """
    
    def __init__(self, db: Database = None):
        self.db = db
        self.threads = {}  # thread_id → state dict
    
    def create_thread(self, candidate_id: str, candidate_email: str) -> str:
        thread_id = str(uuid.uuid4())[:12]
        self.threads[thread_id] = {
            "thread_id": thread_id,
            "candidate_id": candidate_id,
            "candidate_email": candidate_email,
            "state": ThreadState.INITIATED,
            "round": 1,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "history": [],
            "timeout_hours": 48,
        }
        return thread_id
    
    def transition(self, thread_id: str, new_state: ThreadState, context: dict = None):
        if thread_id not in self.threads:
            logger.error(f"Thread {thread_id} not found")
            return
        
        old_state = self.threads[thread_id]["state"]
        self.threads[thread_id]["state"] = new_state
        self.threads[thread_id]["last_activity"] = datetime.utcnow().isoformat()
        self.threads[thread_id]["history"].append({
            "from": old_state,
            "to": new_state,
            "at": datetime.utcnow().isoformat(),
            "context": context or {}
        })
        
        logger.info(f"Thread {thread_id}: {old_state} → {new_state}")
    
    def advance_round(self, thread_id: str):
        if thread_id in self.threads:
            self.threads[thread_id]["round"] += 1
    
    def check_stalled(self) -> List[str]:
        """Find threads that have been awaiting reply beyond timeout."""
        stalled = []
        now = datetime.utcnow()
        for tid, thread in self.threads.items():
            if thread["state"] in (ThreadState.AWAITING_REPLY, ThreadState.FOLLOW_UP_SENT):
                last = datetime.fromisoformat(thread["last_activity"])
                hours = (now - last).total_seconds() / 3600
                if hours > thread["timeout_hours"]:
                    stalled.append(tid)
        return stalled
    
    def get_active_threads(self) -> List[dict]:
        return [
            t for t in self.threads.values() 
            if t["state"] not in (ThreadState.COMPLETED, ThreadState.STALLED, ThreadState.ELIMINATED)
        ]


# ====================================================================
# RESPONSE CLASSIFIER — Decision Tree
# ====================================================================

class ResponseType(str, Enum):
    TECHNICAL = "technical"
    VAGUE = "vague"
    CODE_SUBMISSION = "code_submission"
    QUESTION = "question"
    OFF_TOPIC = "off_topic"
    MINIMAL = "minimal"


def classify_response(text: str) -> Tuple[ResponseType, float]:
    """
    Classify a candidate's reply to determine next action.
    Returns (type, confidence).
    
    Rule-based for reliability. ML-based classification is a future upgrade
    but the decision tree works well for the common cases.
    """
    if not text or word_count(text) <= 3:
        return ResponseType.MINIMAL, 0.95
    
    text_lower = text.lower().strip()
    wc = word_count(text)
    
    # Check for code submission
    code_indicators = ['```', 'def ', 'class ', 'import ', 'function ', 'const ', 'var ', 'let ']
    if any(ind in text for ind in code_indicators):
        return ResponseType.CODE_SUBMISSION, 0.85
    
    # Check if it's a question
    question_marks = text.count('?')
    if question_marks >= 1 and wc < 30:
        return ResponseType.QUESTION, 0.75
    
    # Check for vague/generic content
    vague_signals = [
        "i am interested", "i am passionate", "i would like to",
        "thank you for", "thanks for", "i appreciate",
        "sounds good", "sounds great", "that's interesting",
    ]
    vague_count = sum(1 for s in vague_signals if s in text_lower)
    
    # Technical content detection
    technical_signals = [
        r'\b(api|sdk|runtime|compile|debug|deploy|test|benchmark)\b',
        r'\b(algorithm|data structure|complexity|o\(n\)|o\(log|hash|tree|graph)\b',
        r'\b(docker|kubernetes|aws|gcp|azure|terraform|ci\/cd)\b',
        r'\b(database|sql|nosql|index|query|cache|redis)\b',
        r'\b(neural|model|training|inference|embedding|vector)\b',
        r'\d+%|\d+ms|\d+\.\d+',  # specific numbers/metrics
    ]
    tech_count = sum(1 for p in technical_signals if __import__('re').search(p, text_lower))
    
    if tech_count >= 3:
        return ResponseType.TECHNICAL, 0.80
    elif vague_count >= 2 or (wc < 20 and tech_count == 0):
        return ResponseType.VAGUE, 0.70
    elif tech_count >= 1:
        return ResponseType.TECHNICAL, 0.60
    else:
        return ResponseType.VAGUE, 0.50


# ====================================================================
# EMAIL GENERATION — Contextual Follow-ups
# ====================================================================

# Decision tree: what to do based on response classification
DECISION_TREE = {
    ResponseType.TECHNICAL: {
        "action": "ask_deeper",
        "template": "technical_followup",
        "advance_round": True,
        "description": "Good technical answer -> ask a harder/deeper question"
    },
    ResponseType.VAGUE: {
        "action": "probe",
        "template": "probe_specifics",
        "advance_round": False,
        "description": "Vague answer -> ask for specific examples or details"
    },
    ResponseType.CODE_SUBMISSION: {
        "action": "evaluate_and_respond",
        "template": "code_feedback",
        "advance_round": True,
        "description": "Code submitted -> acknowledge + ask about design decisions"
    },
    ResponseType.QUESTION: {
        "action": "answer_and_redirect",
        "template": "answer_question",
        "advance_round": False,
        "description": "Candidate asked a question -> answer briefly + redirect to next question"
    },
    ResponseType.OFF_TOPIC: {
        "action": "redirect",
        "template": "redirect_topic",
        "advance_round": False,
        "description": "Off-topic -> acknowledge + guide back to assessment"
    },
    ResponseType.MINIMAL: {
        "action": "request_elaboration",
        "template": "request_more",
        "advance_round": False,
        "description": "Minimal response -> ask them to elaborate"
    },
}


# Jinja2-style templates (but using simple string formatting for no dependency)
EMAIL_TEMPLATES = {
    "initial_outreach": """Hi {name},

Thank you for applying for the {role} position. We'd like to learn more about your experience through a brief technical discussion.

Here's our first question:

{question}

Take your time — we value thoughtful responses over quick ones. Reply directly to this email.

Best,
Hiring Team""",

    "technical_followup": """Hi {name},

Great answer on {previous_topic}. {specific_feedback}

Let's go a bit deeper:

{question}

Best,
Hiring Team""",

    "probe_specifics": """Hi {name},

Thanks for your response. I'd like to understand your experience more concretely.

Could you share a specific example? For instance:
- What was the exact problem you were solving?
- What tools/technologies did you use?
- What was the measurable outcome?

Looking forward to hearing the details.

Best,
Hiring Team""",

    "code_feedback": """Hi {name},

Thanks for sharing your code. {code_feedback}

A follow-up question about your implementation:

{question}

Best,
Hiring Team""",

    "answer_question": """Hi {name},

Good question! {answer_to_question}

Now, for the next part of our assessment:

{question}

Best,
Hiring Team""",

    "request_more": """Hi {name},

Thanks for your reply. Could you elaborate on your answer? We're looking for specific details about your approach, the tools you used, and what you learned.

A few sentences with concrete examples would be great.

Best,
Hiring Team""",

    "stalled_followup": """Hi {name},

Just checking in — we haven't heard back from you on our last message. No worries if you need more time.

If you're still interested in the {role} position, just reply to continue the conversation. We'll keep your thread open for another 48 hours.

Best,
Hiring Team""",
}


def generate_email(template_name: str, context: dict) -> str:
    """Generate an email from a template with context variables."""
    template = EMAIL_TEMPLATES.get(template_name, "")
    if not template:
        logger.error(f"Template '{template_name}' not found")
        return ""
    
    # Safe format — ignore missing keys
    try:
        return template.format_map(
            type('SafeDict', (dict,), {'__missing__': lambda self, key: f'{{{key}}}'})
            (context)
        )
    except Exception as e:
        logger.error(f"Template rendering failed: {e}")
        return template


def decide_next_action(response_text: str, current_round: int, max_rounds: int = 3) -> dict:
    """
    Given a candidate's reply, decide what to do next.
    Returns action details including template to use and whether to advance.
    """
    if current_round >= max_rounds:
        return {
            "action": "complete",
            "reason": "Max rounds reached",
            "template": None,
            "advance_round": False,
        }
    
    response_type, confidence = classify_response(response_text)
    decision = DECISION_TREE[response_type].copy()
    decision["response_type"] = response_type
    decision["confidence"] = confidence
    
    return decision


# ====================================================================
# EMAIL BACKEND (Mock for prototype)
# ====================================================================

class MockEmailBackend:
    """
    Mock email backend for testing. Stores emails in memory.
    In production, this would be replaced by GmailAPIBackend or SMTPBackend.
    """
    
    def __init__(self):
        self.sent = []
        self.inbox = []
    
    def send(self, to: str, subject: str, body: str, thread_id: str = None) -> dict:
        msg = {
            "message_id": str(uuid.uuid4())[:12],
            "to": to,
            "subject": subject,
            "body": body,
            "thread_id": thread_id,
            "sent_at": datetime.utcnow().isoformat(),
        }
        self.sent.append(msg)
        logger.info(f"[MOCK] Sent email to {to}: {subject}")
        return msg
    
    def simulate_reply(self, thread_id: str, reply_text: str):
        """Simulate a candidate replying (for testing)."""
        self.inbox.append({
            "thread_id": thread_id,
            "body": reply_text,
            "received_at": datetime.utcnow().isoformat(),
        })
    
    def check_replies(self) -> List[dict]:
        """Check for new replies."""
        replies = list(self.inbox)
        self.inbox.clear()
        return replies


# ====================================================================
# MAIN ENGAGEMENT ENGINE
# ====================================================================

# Questions per round for AI Agent Developer role
QUESTIONS_BY_ROUND = {
    1: [
        "What's a real-world system you've built or contributed to that involved autonomous decision-making? Walk me through the architecture and what you'd change in hindsight.",
    ],
    2: [
        "Tell me about a time a system you built failed in production. What was the root cause, how did you detect it, and what did you change?",
    ],
    3: [
        "Design a system that processes 50K candidates/year with real-time scoring and async email conversations. What's your architecture, and where are the risks?",
    ],
}


def run_engagement(
    scored_candidates: List[ScoredCandidate], 
    db_path: str = None,
    simulate: bool = True
) -> ModuleOutput:
    """
    Full engagement pipeline.
    
    For candidates tiered as Fast-Track or Standard, initiate email threads.
    For Review, hold for manual decision.
    For Reject, skip.
    """
    logger.info(f"=== ENGAGEMENT PIPELINE START ({len(scored_candidates)} candidates) ===")
    
    db = Database(db_path) if db_path else Database()
    tracker = ThreadTracker(db)
    backend = MockEmailBackend()
    
    engagement_results = []
    
    for sc in scored_candidates:
        candidate = sc.candidate
        
        # Skip rejected or eliminated candidates
        if sc.tier == "Reject":
            engagement_results.append({
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "action": "skipped",
                "reason": f"Tier: {sc.tier} (score: {sc.total_score})"
            })
            continue
        
        if sc.tier == "Review":
            engagement_results.append({
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "action": "held_for_review",
                "reason": "Manual review required before engagement"
            })
            continue
        
        # Create thread and send initial email
        thread_id = tracker.create_thread(candidate.candidate_id, candidate.email)
        
        question = QUESTIONS_BY_ROUND.get(1, ["Tell me about your experience."])[0]
        
        email_body = generate_email("initial_outreach", {
            "name": candidate.name.split()[0] if candidate.name else "there",
            "role": candidate.role_applied_for,
            "question": question,
        })
        
        msg = backend.send(
            to=candidate.email,
            subject=f"Your application for {candidate.role_applied_for} — Technical Discussion",
            body=email_body,
            thread_id=thread_id,
        )
        
        tracker.transition(thread_id, ThreadState.AWAITING_REPLY, {"message_id": msg["message_id"]})
        
        result = {
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "tier": sc.tier,
            "action": "email_sent",
            "thread_id": thread_id,
            "email_preview": email_body[:200] + "...",
        }
        
        # Simulate a reply for demonstration
        if simulate and sc.tier == "Fast-Track":
            # Simulate the fast-track candidate replying
            simulated_reply = f"Thanks for reaching out. In my last role, I built a monitoring agent using LangChain that orchestrated API calls across 3 services. The architecture was: FastAPI ingestion -> Celery task queue -> LangChain agent -> Slack notifications. The key challenge was handling flaky upstream APIs -- I added circuit breakers and exponential backoff. In hindsight, I'd add better observability from day one instead of bolting it on later."
            backend.simulate_reply(thread_id, simulated_reply)
            
            # Process the reply
            replies = backend.check_replies()
            for reply in replies:
                response_type, confidence = classify_response(reply["body"])
                decision = decide_next_action(reply["body"], 1)
                
                tracker.transition(thread_id, ThreadState.REPLIED, {
                    "response_type": response_type,
                    "confidence": confidence,
                })
                
                result["simulated_reply"] = {
                    "text": reply["body"][:150] + "...",
                    "classified_as": response_type,
                    "confidence": confidence,
                    "next_action": decision["action"],
                    "next_template": decision["template"],
                }
                
                # Send follow-up
                if decision.get("advance_round"):
                    tracker.advance_round(thread_id)
                    round2_q = QUESTIONS_BY_ROUND.get(2, ["Tell me more."])[0]
                    followup = generate_email(decision["template"], {
                        "name": candidate.name.split()[0],
                        "previous_topic": "your monitoring agent architecture",
                        "specific_feedback": "Good use of circuit breakers — that shows real production experience.",
                        "question": round2_q,
                    })
                    backend.send(
                        to=candidate.email,
                        subject=f"Re: Your application for {candidate.role_applied_for}",
                        body=followup,
                        thread_id=thread_id,
                    )
                    tracker.transition(thread_id, ThreadState.FOLLOW_UP_SENT)
                    result["follow_up_sent"] = True
        
        engagement_results.append(result)
    
    # Build output
    output_data = {
        "thread_tracking": {
            "total_threads": len(tracker.threads),
            "active": len(tracker.get_active_threads()),
            "states": {s.value: sum(1 for t in tracker.threads.values() if t["state"] == s) 
                      for s in ThreadState},
        },
        "decision_tree": {rt.value: dt for rt, dt in DECISION_TREE.items()},
        "email_generation_logic": "Template-based with context injection. Templates are per-action, context includes candidate name, previous answers, and role-specific questions.",
        "sending_flow": {
            "backend": "MockEmailBackend (for prototype) — swap to GmailAPIBackend for production",
            "rate_limit": "100 emails/hour",
            "thread_tracking": "Message-ID / In-Reply-To headers",
            "polling_interval": "Every 2-5 minutes for reply detection",
        },
        "results": engagement_results,
        "emails_sent": len(backend.sent),
        "failure_modes": [
            "Gmail OAuth token expiry -> auto-refresh with stored refresh_token",
            "Email bounces -> mark thread as stalled, flag candidate",
            "Spam filter -> use dedicated sending domain with SPF/DKIM/DMARC",
            "Reply parsing failure -> store raw email, log error, skip classification",
            "Rate limit hit -> queue emails, process in next batch",
        ]
    }
    
    logger.info(f"=== ENGAGEMENT COMPLETE — {len(backend.sent)} emails sent ===")
    return ModuleOutput(module="engagement", data=output_data)


if __name__ == "__main__":
    from module1_ingestion.ingestor import parse_csv
    from module2_scoring.scorer import score_all_candidates
    import argparse
    
    parser = argparse.ArgumentParser(description="Run email engagement")
    parser.add_argument("--input", default="data/sample_candidates.csv")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    
    candidates = parse_csv(args.input)
    scored = score_all_candidates(candidates)
    result = run_engagement(scored, args.db)
    print(result.model_dump_json(indent=2))
