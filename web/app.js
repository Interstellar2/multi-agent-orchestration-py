const API_BASE = window.location.origin.includes('localhost') ? 'http://localhost:8000' : '';
let currentSessionId = localStorage.getItem('current_session_id') || generateId();
let sessions = JSON.parse(localStorage.getItem('sessions') || '[]');
let eventSource = null;

function generateId() {
  return 'sess-' + Math.random().toString(36).slice(2, 10);
}

function saveSessions() {
  localStorage.setItem('sessions', JSON.stringify(sessions));
  localStorage.setItem('current_session_id', currentSessionId);
}

function newSession() {
  currentSessionId = generateId();
  if (!sessions.find(s => s.id === currentSessionId)) {
    sessions.unshift({ id: currentSessionId, name: '新会话 ' + new Date().toLocaleTimeString(), turns: [] });
  }
  saveSessions();
  renderSessionList();
  document.getElementById('chatArea').innerHTML = '';
}

function renderSessionList() {
  const list = document.getElementById('sessionList');
  list.innerHTML = sessions.map(s => `
    <div class="session-item ${s.id === currentSessionId ? 'active' : ''}" onclick="switchSession('${s.id}')">
      <span>${s.name}</span>
      <button class="btn btn-danger btn-sm" onclick="event.stopPropagation();deleteSession('${s.id}')">×</button>
    </div>
  `).join('');
}

async function switchSession(id) {
  currentSessionId = id;
  saveSessions();
  renderSessionList();
  const sess = sessions.find(s => s.id === id);
  const chatArea = document.getElementById('chatArea');
  chatArea.innerHTML = '';
  if (sess && sess.turns && sess.turns.length) {
    sess.turns.forEach(t => appendMessage(t.role, t.content, t.events || []));
  } else if (sess) {
    // 本地没有历史，尝试从后端加载完整对话记录
    await loadSessionDetail(id);
    if (sess.turns && sess.turns.length) {
      sess.turns.forEach(t => appendMessage(t.role, t.content, t.events || []));
    }
  }
}

function deleteSession(id) {
  sessions = sessions.filter(s => s.id !== id);
  if (currentSessionId === id) {
    currentSessionId = sessions.length ? sessions[0].id : generateId();
  }
  saveSessions();
  renderSessionList();
  switchSession(currentSessionId);
}

async function fetchSessions() {
  try {
    const resp = await fetch(`${API_BASE}/api/sessions`);
    if (!resp.ok) return;
    const data = await resp.json();
    const remoteSessions = data.sessions || [];
    let changed = false;
    for (const rs of remoteSessions) {
      const existing = sessions.find(s => s.id === rs.session_id);
      if (!existing) {
        sessions.push({
          id: rs.session_id,
          name: rs.summary ? rs.summary.slice(0, 20) : rs.session_id,
          turns: []
        });
        changed = true;
      }
    }
    if (changed) {
      saveSessions();
      renderSessionList();
    }
  } catch (e) {
    console.error('fetch sessions error', e);
  }
}

async function loadSessionDetail(id) {
  try {
    const resp = await fetch(`${API_BASE}/api/sessions/${id}`);
    if (!resp.ok) return;
    const data = await resp.json();
    const sess = sessions.find(s => s.id === id);
    if (!sess || !data.turns) return;
    // 只保留 human/ai 角色，映射 human -> user，并从 metadata 恢复 events
    sess.turns = data.turns
      .filter(t => t.role === 'human' || t.role === 'ai')
      .map(t => ({
        role: t.role === 'human' ? 'user' : t.role,
        content: t.content,
        events: t.metadata?.events || []
      }));
    saveSessions();
  } catch (e) {
    console.error('load session detail error', e);
  }
}

function appendMessage(role, content, events) {
  const chatArea = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'message message-' + role;
  const eventsHtml = (events || []).map(e => {
    const type = e.type || 'start';
    let text = type;
    if (type === 'intent') text = `意图: ${e.data.domain} (${(e.data.confidence*100).toFixed(0)}%)`;
    if (type === 'semantic') text = `语义: ${e.data.intent}`;
    if (type === 'fastpath') text = `FastPath: ${e.data.domain}`;
    if (type === 'retrieval') text = `检索: ${e.data.count} 条`;
    if (type === 'tool_call') text = `工具: ${e.data.tool}`;
    if (type === 'tool_result') text = `结果: ${e.data.tool}`;
    if (type === 'done') text = '完成';
    if (type === 'error') text = '错误';
    return `<span class="event-tag event-${type}">${text}</span>`;
  }).join('');

  div.innerHTML = `
    <div class="bubble bubble-${role}">
      <div>${escapeHtml(content)}</div>
      ${eventsHtml ? `<div class="events">${eventsHtml}</div>` : ''}
    </div>
  `;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function setLoading(loading) {
  const btn = document.getElementById('sendBtn');
  const input = document.getElementById('userInput');
  btn.disabled = loading;
  input.disabled = loading;
  if (loading) {
    btn.innerHTML = '<span class="spinner"></span>';
  } else {
    btn.textContent = '发送';
  }
}

async function sendMessage() {
  const input = document.getElementById('userInput');
  const query = input.value.trim();
  if (!query) return;
  const mode = document.getElementById('modeSelect').value;

  // 本地保存用户消息
  const sess = sessions.find(s => s.id === currentSessionId);
  if (!sess) {
    sessions.unshift({ id: currentSessionId, name: query.slice(0, 20), turns: [] });
  }
  const currentSess = sessions.find(s => s.id === currentSessionId);
  currentSess.turns.push({ role: 'user', content: query, events: [] });
  saveSessions();
  renderSessionList();

  appendMessage('user', query, []);
  input.value = '';
  setLoading(true);

  // 创建 AI 消息占位
  const aiMsgDiv = appendMessage('ai', '<span class="typing-cursor"></span>', []);
  const aiBubble = aiMsgDiv.querySelector('.bubble-ai');
  let aiText = '';
  const events = [];

  // SSE 请求
  const body = JSON.stringify({ query, mode, session_id: currentSessionId });
  try {
    const resp = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;
        try {
          const evt = JSON.parse(jsonStr);
          events.push(evt);
          if (evt.type === 'chunk') {
            aiText += evt.data.text || '';
            aiBubble.innerHTML = `<div>${escapeHtml(aiText)}<span class="typing-cursor"></span></div><div class="events">${renderEventTags(events)}</div>`;
          } else if (evt.type === 'done') {
            // done 事件的 output 可能只是前 500 字摘要，优先保留已流式积累的全文
            if (!aiText && evt.data.output) aiText = evt.data.output;
            aiBubble.innerHTML = `<div>${escapeHtml(aiText)}</div><div class="events">${renderEventTags(events)}</div>`;
          } else if (evt.type === 'error') {
            aiBubble.innerHTML = `<div style="color:#dc2626">出错: ${escapeHtml(evt.data.error || '')}</div><div class="events">${renderEventTags(events)}</div>`;
          } else {
            // 更新事件标签
            aiBubble.innerHTML = `<div>${escapeHtml(aiText)}<span class="typing-cursor"></span></div><div class="events">${renderEventTags(events)}</div>`;
          }
        } catch (e) { console.error('SSE parse error', e, jsonStr); }
      }
    }

    // 保存 AI 回复到本地
    currentSess.turns.push({ role: 'ai', content: aiText, events });
    saveSessions();
  } catch (err) {
    aiBubble.innerHTML = `<div style="color:#dc2626">请求失败: ${escapeHtml(err.message)}</div>`;
  } finally {
    setLoading(false);
  }
}

function renderEventTags(events) {
  return events.map(e => {
    const type = e.type || 'start';
    let text = type;
    if (type === 'intent') text = `意图: ${e.data.domain}`;
    if (type === 'semantic') text = `语义: ${e.data.intent}`;
    if (type === 'fastpath') text = `FastPath`;
    if (type === 'retrieval') text = `检索 ${e.data.count} 条`;
    if (type === 'tool_call') text = `工具: ${e.data.tool}`;
    if (type === 'tool_result') text = `结果: ${e.data.tool}`;
    if (type === 'done') text = '完成';
    if (type === 'error') text = '错误';
    return `<span class="event-tag event-${type}">${text}</span>`;
  }).join('');
}

async function init() {
  if (!sessions.find(s => s.id === currentSessionId)) {
    sessions.unshift({ id: currentSessionId, name: '新会话', turns: [] });
    saveSessions();
  }
  renderSessionList();
  await fetchSessions();
  // 如果当前 session 在服务端有记录但本地无 turns，自动加载详情
  const sess = sessions.find(s => s.id === currentSessionId);
  if (sess && (!sess.turns || !sess.turns.length)) {
    await loadSessionDetail(currentSessionId);
    const chatArea = document.getElementById('chatArea');
    chatArea.innerHTML = '';
    if (sess.turns && sess.turns.length) {
      sess.turns.forEach(t => appendMessage(t.role, t.content, t.events || []));
    }
  }
}

init();