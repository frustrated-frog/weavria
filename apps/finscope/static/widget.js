const state = {
  context: null,
  busy: false,
  clarificationId: null,
  sessionId: window.localStorage.getItem('weavria.marketPulse.sessionId')
};
const markdownReady = ensureMarkdownRenderer();

const conversation = document.querySelector('#conversation');
const form = document.querySelector('#composer');
const question = document.querySelector('#question');
const send = document.querySelector('#send');
const status = document.querySelector('#agent-status');
const contextLabel = document.querySelector('#context-label');

window.addEventListener('message', (event) => {
  if (event.source !== window.parent || !isContextMessage(event.data)) return;
  state.context = event.data.context;
  const date = state.context.business_date || '最新交易日';
  contextLabel.textContent = `${date} · ${viewLabel(state.context.current_view)}`;
});

window.parent.postMessage({ type: 'weavria:ready', version: 1 }, '*');

document.querySelectorAll('.suggestions button').forEach((button) => {
  button.addEventListener('click', () => {
    question.value = button.textContent;
    question.focus();
  });
});

question.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    void submitQuestion();
  }
});

send.addEventListener('click', () => {
  void submitQuestion();
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  void submitQuestion();
});

async function submitQuestion() {
  const text = question.value.trim();
  if (!text || state.busy) return;
  if (!state.context) {
    setStatus('等待页面', 'error');
    return;
  }
  question.value = '';
  appendMessage('user', text);
  setBusy(true);
  try {
    const result = await sendMessage(text, false);
    state.clarificationId = null;
    await markdownReady;
    appendMessage('agent', result.answer || result.error || '本次运行没有生成可交付答案。', result.tools, result.clarification_id);
    const completedLabel = result.task_outcome === 'verified'
      ? '已核验'
      : result.task_outcome === 'unverifiable'
        ? '未能核验'
        : '已完成';
    setStatus(result.status === 'completed' ? completedLabel : result.status, result.status === 'completed' ? '' : 'error');
  } catch (error) {
    await markdownReady;
    appendMessage('agent', error instanceof Error ? error.message : 'Agent 暂时不可用。');
    setStatus('连接失败', 'error');
  } finally {
    setBusy(false, false);
  }
}

function ensureMarkdownRenderer() {
  if (window.WeavriaMarkdown) return Promise.resolve(true);
  return new Promise((resolve) => {
    const script = document.createElement('script');
    script.src = '/widget-assets/markdown.js';
    script.addEventListener('load', () => resolve(Boolean(window.WeavriaMarkdown)), {
      once: true
    });
    script.addEventListener('error', () => resolve(false), { once: true });
    document.head.append(script);
  });
}

async function sendMessage(text, retried) {
  if (!state.sessionId) {
    const created = await request('/api/sessions', { method: 'POST' });
    state.sessionId = created.session_id;
    window.localStorage.setItem('weavria.marketPulse.sessionId', state.sessionId);
  }
  try {
    return await request(`/api/sessions/${encodeURIComponent(state.sessionId)}/messages`, {
      method: 'POST',
      body: JSON.stringify({
        input_id: crypto.randomUUID(),
        text,
        clarification_id: state.clarificationId,
        context: state.context
      })
    });
  } catch (error) {
    if (!retried && error instanceof HttpError && error.status === 404) {
      state.sessionId = null;
      window.localStorage.removeItem('weavria.marketPulse.sessionId');
      return sendMessage(text, true);
    }
    throw error;
  }
}

async function request(path, options) {
  const response = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }
  });
  let body = null;
  try { body = await response.json(); } catch (_) { /* safe empty error */ }
  if (!response.ok) throw new HttpError(response.status, body?.detail || `请求失败（${response.status}）`);
  return body;
}

function appendMessage(role, text, tools = [], clarificationId = null) {
  const item = document.createElement('article');
  item.className = `message ${role}`;
  const label = document.createElement('span');
  label.textContent = role === 'user' ? '你的问题' : 'AGENT 回答';
  const content = document.createElement(role === 'agent' ? 'div' : 'p');
  if (role === 'agent' && window.WeavriaMarkdown) {
    content.className = 'markdown';
    window.WeavriaMarkdown.renderMarkdown(content, text);
  } else {
    content.textContent = text;
  }
  item.append(label, content);
  if (tools.length) {
    const trace = document.createElement('div');
    trace.className = 'trace';
    tools.forEach((tool) => {
      const chip = document.createElement('i');
      chip.className = tool.status;
      chip.textContent = `${tool.tool_name} · ${tool.status === 'succeeded' ? '完成' : '失败'}`;
      trace.append(chip);
    });
    item.append(trace);
  }
  if (clarificationId) {
    const reply = document.createElement('button');
    reply.type = 'button';
    reply.textContent = '回答这个追问';
    reply.addEventListener('click', () => {
      state.clarificationId = clarificationId;
      question.placeholder = '输入对这个追问的回答…';
      question.focus();
      setStatus('正在回答追问', '');
    });
    item.append(reply);
  }
  conversation.append(item);
  conversation.scrollTop = conversation.scrollHeight;
}

function setBusy(busy, updateStatus = true) {
  state.busy = busy;
  send.disabled = busy;
  question.disabled = busy;
  if (busy && updateStatus) setStatus('正在取证', 'busy');
}

function setStatus(text, tone) {
  status.textContent = text;
  status.className = `agent-status${tone ? ` ${tone}` : ''}`;
}

function isContextMessage(value) {
  return value && value.type === 'weavria:context' && value.context?.schema_version === 1
    && value.context.surface === 'market_pulse';
}

function viewLabel(view) {
  return ({ panorama: '市场全景', review: '今日雷达', transition: '转折与情景', breadth: '市场宽度', history: '历史演变' })[view] || '市场状态';
}

class HttpError extends Error {
  constructor(statusCode, message) {
    super(message);
    this.status = statusCode;
  }
}
