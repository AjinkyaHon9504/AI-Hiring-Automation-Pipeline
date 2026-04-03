document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const runBtn = document.getElementById('runPipelineBtn');
    const uploadInput = document.getElementById('csvUpload');
    const statusDot = document.getElementById('pipelineStatusDot');
    const statusText = document.getElementById('pipelineStatusText');
    const progressContainer = document.getElementById('pipelineProgress');
    const progressBar = document.getElementById('progressBar');
    
    // Polling interval
    let pollInterval = null;

    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            // Update active states
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            // Switch tabs
            const targetTab = item.getAttribute('data-tab');
            document.querySelectorAll('.tab-pane').forEach(t => t.classList.remove('active'));
            document.getElementById(`tab-${targetTab}`).classList.add('active');
            
            // Update title
            document.getElementById('currentTabTitle').textContent = 
                targetTab.charAt(0).toUpperCase() + targetTab.slice(1).replace('-', ' ') + ' Dashboard';
        });
    });

    // Run Pipeline
    runBtn.addEventListener('click', () => {
        if(runBtn.disabled) return;
        
        fetch('/api/run-sample', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if(data.status === 'started') {
                    startPolling();
                }
            })
            .catch(err => console.error(err));
    });

    // Upload CSV
    uploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if(!file) return;

        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/run', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if(data.status === 'started') {
                startPolling();
            }
        })
        .catch(err => console.error(err));
    });

    // Polling Logic
    function startPolling() {
        runBtn.disabled = true;
        runBtn.classList.remove('pulse');
        progressContainer.classList.remove('hidden');
        
        // Reset steps
        document.querySelectorAll('.step').forEach(s => {
            s.classList.remove('active', 'completed');
        });

        if(pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(fetchStatus, 1000);
        fetchStatus();
    }

    function fetchStatus() {
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                updateStatusUI(data);
                
                if (data.status === 'completed' || data.status === 'error') {
                    clearInterval(pollInterval);
                    runBtn.disabled = false;
                    runBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Run Again';
                    
                    if(data.status === 'completed') {
                        setTimeout(() => progressContainer.classList.add('hidden'), 5000);
                        refreshAllData();
                    }
                }
            })
            .catch(err => console.error("Poll error:", err));
    }

    function updateStatusUI(data) {
        // Dot and text
        statusDot.className = `status-dot ${data.status}`;
        statusText.textContent = `System ${data.status.charAt(0).toUpperCase() + data.status.slice(1)}`;
        
        // Progress bar
        progressBar.style.width = `${data.progress}%`;
        
        // Steps
        data.steps_completed.forEach(step => {
            const el = document.getElementById(`step-${step}`);
            if(el) el.classList.add('completed');
        });
        
        if (data.current_step && data.current_step !== 'done') {
            const currentEl = document.getElementById(`step-${data.current_step}`);
            if(currentEl) currentEl.classList.add('active');
            statusText.textContent = `Running: ${data.current_step}...`;
        }
    }

    function refreshAllData() {
        Promise.all([
            fetch('/api/scores').then(r => r.json()),
            fetch('/api/anticheat').then(r => r.json()),
            fetch('/api/engagement').then(r => r.json()),
            fetch('/api/learning').then(r => r.json())
        ]).then(([scores, anticheat, engagement, learning]) => {
            updateDashboard(scores);
            updateCandidatesTable(scores, anticheat);
            updateAntiCheatView(anticheat, scores);
            updateEngagementView(engagement);
            updateLearningView(learning);
        });
    }

    function updateDashboard(scores) {
        if(!scores || scores.length === 0) return;

        // Ensure scores are sorted descending
        scores.sort((a,b) => b.total_score - a.total_score);

        document.getElementById('statTotalCandidates').textContent = scores.length;
        document.getElementById('statFastTrack').textContent = scores.filter(s => s.tier === 'Fast-Track').length;
        document.getElementById('statRejected').textContent = scores.filter(s => s.tier === 'Reject').length;
        // Mock flag count based on tier or actual flags if parsed (we'll estimate for now)
        document.getElementById('statFlags').textContent = scores.reduce((acc, s) => acc + (s.flags ? s.flags.length : 0), 0);

        // Top Candidates list
        const topList = document.getElementById('topCandidatesList');
        topList.innerHTML = '';
        
        scores.slice(0, 5).forEach((s, idx) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <div class="cand-info">
                    <h4>#${idx+1} ${s.name}</h4>
                    <p>Score: ${s.total_score.toFixed(1)} | Tech: ${s.dimension_scores.technical_relevance.toFixed(0)}</p>
                </div>
                <span class="badge ${s.tier.toLowerCase()}">${s.tier}</span>
            `;
            li.onclick = () => showCandidateModal(s.candidate_id);
            topList.appendChild(li);
        });
        
        // Simple manual insight push
        const insights = document.getElementById('dashboardInsights');
         insights.innerHTML = `
            <div class="insight-card">
                <h4>Hiring Bar Analytics</h4>
                <p>Top candidate ${scores[0].name} set a new high water mark (${scores[0].total_score.toFixed(1)}). The overall pool skews technical.</p>
            </div>
            <div class="insight-card">
                <h4>Rejection Causes</h4>
                <p>Primary rejection cause this batch: Lack of specificity (generic phrases).</p>
            </div>
        `;
    }

    function updateCandidatesTable(scores, anticheat) {
        const tbody = document.getElementById('candidatesTableBody');
        tbody.innerHTML = '';
        
        const filter = document.getElementById('tierFilter').value;
        
        scores.forEach(s => {
            if(filter !== 'all' && s.tier !== filter) return;
            
            const tr = document.createElement('tr');
            
            const flagCount = (s.flags ? s.flags.length : 0);
            const flagHTML = flagCount > 0 ? `<span class="text-red"><i class="fa-solid fa-flag"></i> ${flagCount}</span>` : `<span class="text-secondary">-</span>`;
            
            tr.innerHTML = `
                <td><strong>${s.name}</strong><br><small class="text-secondary">${s.email}</small></td>
                <td><strong class="${s.total_score > 80 ? 'text-green' : (s.total_score < 50 ? 'text-red' : '')}">${s.total_score.toFixed(1)}</strong></td>
                <td><span class="badge ${s.tier.toLowerCase()}">${s.tier}</span></td>
                <td>${s.dimension_scores.technical_relevance.toFixed(0)}</td>
                <td>${flagHTML}</td>
                <td><button class="btn btn-primary" onclick="showCandidateModal('${s.candidate_id}')">View</button></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Bind filter Native JS way
    document.getElementById('tierFilter').addEventListener('change', () => {
        fetch('/api/scores').then(r => r.json())
            .then(scores => updateCandidatesTable(scores, null));
    });

    function updateAntiCheatView(ac, scores) {
        const container = document.getElementById('antiCheatContent');
        if(!ac || !ac.similar_pairs) return;
        
        let html = `
            <div class="stats-grid">
                <div class="stat-card glass-panel" style="padding:15px">
                    <div>
                        <span class="stat-label">Analysis Method</span>
                        <h4 class="text-purple">${ac.method.toUpperCase()}</h4>
                    </div>
                </div>
                <div class="stat-card glass-panel" style="padding:15px">
                    <div>
                        <span class="stat-label">Similar Pairs Detected</span>
                        <h4 class="text-warning">${ac.similar_pairs.length}</h4>
                    </div>
                </div>
            </div>
        `;

        // Similar Pairs
        if(ac.similar_pairs.length > 0) {
            html += `<h4><i class="fa-solid fa-copy text-warning"></i> Copy Detection</h4>
            <div class="candidate-list" style="margin-top:10px; margin-bottom: 30px;">`;
            
            ac.similar_pairs.forEach(pair => {
                html += `
                <li style="flex-direction:column; align-items:flex-start; border-left: 3px solid var(--warning)">
                    <div style="display:flex; justify-content:space-between; width:100%; margin-bottom:10px;">
                        <strong>${pair.candidate_a_name} ↔ ${pair.candidate_b_name}</strong>
                        <span class="badge review">Sim: ${(pair.similarity*100).toFixed(1)}%</span>
                    </div>
                    <div style="display:flex; gap:20px; font-size:0.85rem; color:var(--text-secondary); width:100%">
                        <div style="flex:1; background:rgba(0,0,0,0.2); padding:10px; border-radius:4px;">
                            <em>${pair.candidate_a_name}</em>: "${pair.text_a_preview}..."
                        </div>
                        <div style="flex:1; background:rgba(0,0,0,0.2); padding:10px; border-radius:4px;">
                             <em>${pair.candidate_b_name}</em>: "${pair.text_b_preview}..."
                        </div>
                    </div>
                </li>`;
            });
            html += `</div>`;
        }

        // Timing Anomalies
        let timingAnomalies = [];
        Object.keys(ac.timing || {}).forEach(cid => {
            if(ac.timing[cid].flags && ac.timing[cid].flags.length > 0) {
                const name = scores.find(s => s.candidate_id === cid)?.name || cid;
                timingAnomalies.push({name, flags: ac.timing[cid].flags});
            }
        });

        if(timingAnomalies.length > 0) {
            html += `<h4><i class="fa-solid fa-stopwatch text-red"></i> Timing Anomalies</h4>
             <ul class="candidate-list" style="margin-top:10px;">`;
             timingAnomalies.forEach(t => {
                 html += `<li>
                    <span><strong>${t.name}</strong></span>
                    <span class="text-red">${t.flags.join(", ")}</span>
                 </li>`;
             });
             html += `</ul>`;
        } else {
             html += `<p class="text-success"><i class="fa-solid fa-check"></i> No timing anomalies detected.</p>`;
        }

        container.innerHTML = html;
    }

    function updateEngagementView(eng) {
        const container = document.getElementById('engagementContent');
        if(!eng || !eng.results) return;

        let sent = eng.emails_sent || 0;
        let html = `
             <div class="stats-grid">
                <div class="stat-card glass-panel" style="padding:15px">
                    <div class="stat-icon text-primary"><i class="fa-solid fa-paper-plane"></i></div>
                    <div>
                        <span class="stat-label">Emails Sent</span>
                        <h4 class="">${sent}</h4>
                    </div>
                </div>
                 <div class="stat-card glass-panel" style="padding:15px">
                    <div class="stat-icon text-success"><i class="fa-solid fa-reply-all"></i></div>
                    <div>
                        <span class="stat-label">Simulated Replies</span>
                        <h4 class="">1</h4>
                    </div>
                </div>
            </div>

            <h4 style="margin-top:20px; margin-bottom:15px;">Engagement Threads</h4>
            <div class="candidate-list">
        `;

        Object.keys(eng.results).forEach(cid => {
            const thread = eng.results[cid];
            html += `
                <li style="flex-direction:column; align-items:flex-start">
                    <div style="display:flex; justify-content:space-between; width:100%; margin-bottom:5px;">
                        <strong>Thread ID: ${thread.thread_id.substring(0,8)}...</strong>
                        <span class="badge ${thread.state === 'completed' ? 'fast-track' : 'standard'}">${thread.state}</span>
                    </div>
                    <div style="font-size:0.85rem; color:var(--text-secondary); width:100%">
                        <p>Round: ${thread.current_round} | Interactions: ${thread.history.length}</p>
                    </div>
                </li>
            `;
        });

        html += `</div>`;
        container.innerHTML = html;
    }

    function updateLearningView(lrn) {
        const container = document.getElementById('learningContent');
        if(!lrn) return;
        
        let recs = lrn.weight_recommendations || {};
        let dist = lrn.distribution?.score_stats || {};
        
        let html = `
            <div class="stats-grid">
                <div class="stat-card glass-panel" style="padding:15px">
                    <div><span class="stat-label">Mean Score</span><h4>${(dist.mean||0).toFixed(1)}</h4></div>
                </div>
                <div class="stat-card glass-panel" style="padding:15px">
                     <div><span class="stat-label">Median Score</span><h4>${(dist.median||0).toFixed(1)}</h4></div>
                </div>
                 <div class="stat-card glass-panel" style="padding:15px">
                     <div><span class="stat-label">Max Score</span><h4>${(dist.max||0).toFixed(1)}</h4></div>
                </div>
            </div>
            
            <h4 style="margin: 20px 0 10px;">Recommended Weight Adjustments</h4>
        `;
        
        if (recs.suggested_adjustments && Object.keys(recs.suggested_adjustments).length > 0) {
             html += `<table class="data-table">
                <thead><tr><th>Dimension</th><th>Suggested ∆</th></tr></thead><tbody>`;
             
             Object.keys(recs.suggested_adjustments).forEach(dim => {
                 let val = recs.suggested_adjustments[dim];
                 let color = val > 0 ? 'text-green' : 'text-red';
                 let sign = val > 0 ? '+' : '';
                 html += `<tr>
                    <td>${dim.replace('_', ' ')}</td>
                    <td class="${color}">${sign}${val.toFixed(4)}</td>
                 </tr>`;
             });
             html += `</tbody></table>`;
             
             if(recs.reasoning) {
                 html += `<div style="margin-top:15px; padding:15px; background:rgba(0,0,0,0.2); border-radius:8px;">
                     <strong>Reasoning:</strong><br>
                     <p class="text-secondary" style="margin-top:5px;">${recs.reasoning.join('<br>')}</p>
                 </div>`;
             }
        } else {
             html += `<p class="empty-state">No adjustments recommended based on current distribution.</p>`;
        }
        
        container.innerHTML = html;
    }

    // Modal Logic attached to window so inline onclick works
    window.showCandidateModal = function(candidateId) {
        fetch(`/api/candidate/${candidateId}`)
            .then(res => res.json())
            .then(data => {
                const s = data.scores;
                const c = data.candidate;
                const modal = document.getElementById('candidateModal');
                
                document.getElementById('modalCandidateName').textContent = s.name;
                
                const tierBadge = document.getElementById('modalCandidateTier');
                tierBadge.textContent = s.tier;
                tierBadge.className = `badge ${s.tier.toLowerCase()}`;
                
                let dims = s.dimension_scores;
                let html = `
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2>Total: <span class="${s.total_score > 80 ? 'text-green' : (s.total_score < 50 ? 'text-red' : '')}">${s.total_score.toFixed(1)}</span> / 100</h2>
                        <div style="display:flex; gap:10px;">
                            ${s.github_url ? `<a href="${s.github_url}" target="_blank" style="color:var(--text-secondary)"><i class="fa-brands fa-github"></i> Profile</a>` : ''}
                        </div>
                    </div>
                    
                    <div class="score-breakdown">
                        <div class="score-item"><span>Technical</span><strong>${dims.technical_relevance.toFixed(0)}</strong></div>
                        <div class="score-item"><span>Quality</span><strong>${dims.answer_quality.toFixed(0)}</strong></div>
                        <div class="score-item"><span>Credibility</span><strong>${dims.profile_credibility.toFixed(0)}</strong></div>
                        <div class="score-item"><span>Specificity</span><strong>${dims.specificity.toFixed(0)}</strong></div>
                        <div class="score-item"><span>Timing</span><strong>${dims.timing.toFixed(0)}</strong></div>
                    </div>
                    
                    <div class="explanations" style="margin-bottom:20px;">
                        <h4>Scoring Breakdown</h4>
                        <ul>
                            ${s.explanation.map(e => `<li>${e}</li>`).join('')}
                        </ul>
                    </div>
                `;
                
                if (data.similarity_flags && data.similarity_flags.length > 0) {
                     html += `<div style="padding:15px; background:var(--warning-bg); border-left:4px solid var(--warning); border-radius:4px; margin-bottom:20px;">
                        <h4 class="text-warning"><i class="fa-solid fa-triangle-exclamation"></i> Copy Flag</h4>
                        <p>High similarity detected with another candidate (${(data.similarity_flags[0].similarity*100).toFixed(1)}%)</p>
                     </div>`;
                }
                
                html += `<h4>Answers</h4>
                <div style="display:flex; flex-direction:column; gap:15px; margin-top:10px;">`;
                
                s.answers.forEach((ans, i) => {
                    html += `
                        <div style="background:rgba(255,255,255,0.03); padding:15px; border-radius:8px; border:1px solid var(--border-color);">
                            <div style="font-weight:600; margin-bottom:8px; color:var(--text-secondary)">Q: ${ans.question_id}</div>
                            <p style="font-family:monospace; font-size:0.9rem; white-space:pre-wrap;">${ans.answer_text || '<span class="text-muted">No answer</span>'}</p>
                        </div>
                    `;
                });
                
                html += `</div>`;
                
                document.getElementById('modalCandidateDetails').innerHTML = html;
                modal.classList.add('show');
            });
    }

    document.getElementById('closeModal').addEventListener('click', () => {
        document.getElementById('candidateModal').classList.remove('show');
    });
});
