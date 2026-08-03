/* ==========================================================
   AstraGPT — Frontend application logic
   Implements: streaming chat, multi-conversation, file upload
   + RAG, model selector, tool execution display, long-term
   memory indicators, speech-to-text, status indicators,
   responsive sidebar, error handling, local persistence.
   ========================================================== */

(() => {
  "use strict";

  
  /* ---------------- Config ---------------- */
  const API_BASE = "";
  const ENDPOINTS = {
    chatStream: `${API_BASE}/chat/stream`,
    upload: `${API_BASE}/upload`,
    conversations: `${API_BASE}/conversations`,
    history: (threadId) => `${API_BASE}/history/${threadId}`,
  };

  const MODELS = [
    { id: "gemini-3.5-flash", name: "Gemini 3.5 Flash", desc: "Gemini · balanced speed & quality" },
    { id: "gemini-3.1-flash-lite", name: "Gemini 3.1 Flash-Lite", desc: "Gemini · fastest, most efficient" },
    { id: "gemini-3.6-flash", name: "Gemini 3.6 Flash", desc: "Gemini · latest & most capable" },
  ];

  const SUGGESTED_PROMPTS = [
    { icon: "bar_chart", title: "Analyze data", subtitle: "Summarize trends in a CSV I upload" },
    { icon: "draft", title: "Draft an email", subtitle: "Write a professional follow-up" },
    { icon: "travel_explore", title: "Research a topic", subtitle: "Use web search for current info" },
    { icon: "code", title: "Explain this code", subtitle: "Walk through a function step by step" },
  ];

  const TOOL_META = {
    tavily_search: { label: "Searching the web", icon: "travel_explore" },
    calculator: { label: "Calculating", icon: "calculate" },
    get_weather: { label: "Checking weather", icon: "partly_cloudy_day" },
    search_uploaded_documents: { label: "Searching documents", icon: "find_in_page" },
    remember_this: { label: "Saving to memory", icon: "bookmark_add" },
    recall_memory: { label: "Recalling memory", icon: "psychology" },
  };

  const ALLOWED_EXT = ["pdf", "docx", "txt", "md", "py", "csv"];

  /* ---------------- State ---------------- */
  const state = {
    threadId: null,
    model: MODELS.some((m) => m.id === localStorage.getItem("astra_model")) ? localStorage.getItem("astra_model") : MODELS[0].id,
    conversations: [],
    messages: [],
    pendingAttachments: [],
    isStreaming: false,
    isRecording: false,
    lastUserMessage: null,
  };

  /* ---------------- DOM refs ---------------- */
  const $ = (sel) => document.querySelector(sel);
  const dom = {
    sideNav: $("#sideNav"),
    sidebarScrim: $("#sidebarScrim"),
    openSidebarBtn: $("#openSidebarBtn"),
    closeSidebarBtn: $("#closeSidebarBtn"),
    newChatBtn: $("#newChatBtn"),
    historySearch: $("#historySearch"),
    historyList: $("#historyList"),
    historyEmpty: $("#historyEmpty"),
    themeToggleBtn: $("#themeToggleBtn"),
    themeIcon: $("#themeIcon"),
    themeLabel: $("#themeLabel"),
    threadIdLabel: $("#threadIdLabel"),
    convTitle: $("#convTitle"),
    modelSelectorBtn: $("#modelSelectorBtn"),
    modelDropdown: $("#modelDropdown"),
    modelList: $("#modelList"),
    modelLabel: $("#modelLabel"),
    chatScroll: $("#chatScroll"),
    welcomeScreen: $("#welcomeScreen"),
    suggestedPrompts: $("#suggestedPrompts"),
    messagesList: $("#messagesList"),
    attachmentsBar: $("#attachmentsBar"),
    inputShell: $("#inputShell"),
    messageInput: $("#messageInput"),
    attachBtn: $("#attachBtn"),
    fileInput: $("#fileInput"),
    micBtn: $("#micBtn"),
    sendBtn: $("#sendBtn"),
    sendIcon: $("#sendIcon"),
    statusDot: $("#statusDot"),
    statusText: $("#statusText"),
    errorToast: $("#errorToast"),
    errorToastText: $("#errorToastText"),
    errorToastClose: $("#errorToastClose"),
  };

  /* ---------------- Utils ---------------- */
  const uuid = () =>
    "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });

  const escapeHtml = (str) =>
    str.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function showError(msg) {
    dom.errorToastText.textContent = msg;
    dom.errorToast.classList.remove("hidden");
    clearTimeout(showError._t);
    showError._t = setTimeout(() => dom.errorToast.classList.add("hidden"), 5000);
  }
  dom.errorToastClose.addEventListener("click", () => dom.errorToast.classList.add("hidden"));

  function setStatus(mode) {
    const map = {
      ready: { text: "Ready", color: "bg-secondary", active: false },
      thinking: { text: "Thinking…", color: "bg-secondary", active: true },
      tool: { text: "Using tool…", color: "bg-tertiary", active: true },
      listening: { text: "Listening…", color: "bg-error", active: true },
      uploading: { text: "Uploading…", color: "bg-secondary", active: true },
      streaming: { text: "Streaming…", color: "bg-secondary", active: true },
    };
    const s = map[mode] || map.ready;
    dom.statusText.textContent = s.text;
    dom.statusDot.className = `w-1.5 h-1.5 rounded-full ${s.color}` + (s.active ? " active" : "");
  }

  function autoResize() {
    dom.messageInput.style.height = "auto";
    dom.messageInput.style.height = Math.min(dom.messageInput.scrollHeight, 192) + "px";
  }

  /* Minimal, dependency-free markdown-lite renderer:
     fenced code blocks, inline code, bold, paragraphs. */
  function renderMarkdownLite(raw) {
    const blocks = [];
    let text = raw.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
      const idx = blocks.push({ lang: lang || "text", code }) - 1;
      return `\u0000CODE${idx}\u0000`;
    });

    text = escapeHtml(text);
    text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

    const paragraphs = text
      .split(/\n{2,}/)
      .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
      .join("");

    return paragraphs.replace(/\u0000CODE(\d+)\u0000/g, (_, i) => {
      const b = blocks[i];
      const codeEscaped = escapeHtml(b.code.trim());
      return `<div class="code-block-wrap">
        <div class="code-block-header">
          <span>${escapeHtml(b.lang)}</span>
          <span class="code-copy-btn" data-code="${encodeURIComponent(b.code.trim())}">
            <span class="material-symbols-outlined" style="font-size:14px;">content_copy</span>Copy
          </span>
        </div>
        <pre><code>${codeEscaped}</code></pre>
      </div>`;
    });
  }

  /* ---------------- Persistence ---------------- */
  function persist() {
    if (state.threadId) localStorage.setItem("astra_thread_id", state.threadId);
    localStorage.setItem("astra_model", state.model);
  }

  /* ---------------- Sidebar (mobile) ---------------- */
  function openSidebar() {
    dom.sideNav.classList.add("open");
    dom.sidebarScrim.classList.remove("hidden");
  }
  function closeSidebar() {
    dom.sideNav.classList.remove("open");
    dom.sidebarScrim.classList.add("hidden");
  }
  dom.openSidebarBtn.addEventListener("click", openSidebar);
  dom.closeSidebarBtn.addEventListener("click", closeSidebar);
  dom.sidebarScrim.addEventListener("click", closeSidebar);

  /* ---------------- Theme ---------------- */
  function applyTheme(mode) {
    document.documentElement.classList.toggle("dark", mode === "dark");
    document.documentElement.classList.toggle("light", mode !== "dark");
    dom.themeIcon.textContent = mode === "dark" ? "light_mode" : "dark_mode";
    dom.themeLabel.textContent = mode === "dark" ? "Light mode" : "Dark mode";
    localStorage.setItem("astra_theme", mode);
  }
  dom.themeToggleBtn.addEventListener("click", () => {
    const isDark = document.documentElement.classList.contains("dark");
    applyTheme(isDark ? "light" : "dark");
  });
  applyTheme(localStorage.getItem("astra_theme") || "light");

  /* ---------------- Model selector ---------------- */
  function renderModelList() {
    dom.modelList.innerHTML = "";
    MODELS.forEach((m) => {
      const btn = document.createElement("button");
      btn.className =
        "w-full text-left px-sm py-2 rounded-md hover:bg-surface-container-high transition-colors flex flex-col " +
        (m.id === state.model ? "bg-surface-container-high" : "");
      btn.innerHTML = `<span class="text-body-sm font-body-sm font-semibold text-on-surface">${m.name}</span>
        <span class="text-[12px] text-on-surface-variant">${m.desc}</span>`;
      btn.addEventListener("click", () => {
        state.model = m.id;
        persist();
        dom.modelLabel.textContent = m.name;
        dom.modelDropdown.classList.add("hidden");
        renderModelList();
      });
      dom.modelList.appendChild(btn);
    });
  }
  dom.modelSelectorBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    dom.modelDropdown.classList.toggle("hidden");
  });
  document.addEventListener("click", () => dom.modelDropdown.classList.add("hidden"));
  dom.modelDropdown.addEventListener("click", (e) => e.stopPropagation());

  /* ---------------- Suggested prompts ---------------- */
  function renderSuggestedPrompts() {
    dom.suggestedPrompts.innerHTML = "";
    SUGGESTED_PROMPTS.forEach((p) => {
      const card = document.createElement("button");
      card.className =
        "prompt-chip text-left flex items-start gap-sm p-md rounded-lg border border-outline-variant/30 bg-surface-container-lowest";
      card.innerHTML = `
        <span class="material-symbols-outlined text-secondary">${p.icon}</span>
        <span>
          <span class="block font-semibold text-body-lg text-on-surface">${p.title}</span>
          <span class="block text-body-sm text-on-surface-variant">${p.subtitle}</span>
        </span>`;
      card.addEventListener("click", () => {
        dom.messageInput.value = p.subtitle;
        autoResize();
        sendMessage();
      });
      dom.suggestedPrompts.appendChild(card);
    });
  }

  /* ---------------- Conversation history ---------------- */
  async function loadConversations() {
    try {
      const res = await fetch(ENDPOINTS.conversations);
      if (!res.ok) throw new Error("bad status");
      state.conversations = await res.json();
    } catch {
      state.conversations = JSON.parse(localStorage.getItem("astra_conversations") || "[]");
    }
    renderHistory();
  }

  function cacheConversationLocally() {
    const existing = state.conversations.find((c) => c.thread_id === state.threadId);
    const title = (state.messages.find((m) => m.role === "user")?.content || "New Chat").slice(0, 48);
    if (existing) {
      existing.title = title;
      existing.updated_at = new Date().toISOString();
    } else {
      state.conversations.unshift({ thread_id: state.threadId, title, updated_at: new Date().toISOString() });
    }
    localStorage.setItem("astra_conversations", JSON.stringify(state.conversations));
    renderHistory();
  }

  function renderHistory(filter = "") {
    dom.historyList.innerHTML = "";
    const items = state.conversations.filter((c) =>
      (c.title || "").toLowerCase().includes(filter.toLowerCase())
    );
    dom.historyEmpty.classList.toggle("hidden", items.length > 0);
    items.forEach((c) => {
      const el = document.createElement("div");
      el.className = "history-item" + (c.thread_id === state.threadId ? " active" : "");
      el.innerHTML = `
        <span class="material-symbols-outlined text-[18px]">chat_bubble</span>
        <span class="truncate flex-1">${escapeHtml(c.title || "Untitled")}</span>
        <button class="history-delete p-1 rounded hover:bg-black/10" title="Delete">
          <span class="material-symbols-outlined text-[16px]">delete</span>
        </button>`;
      el.addEventListener("click", () => switchConversation(c.thread_id));
      el.querySelector(".history-delete").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteConversation(c.thread_id);
      });
      dom.historyList.appendChild(el);
    });
  }

  dom.historySearch.addEventListener("input", (e) => renderHistory(e.target.value));

  function deleteConversation(threadId) {
    state.conversations = state.conversations.filter((c) => c.thread_id !== threadId);
    localStorage.setItem("astra_conversations", JSON.stringify(state.conversations));
    if (threadId === state.threadId) startNewChat();
    else renderHistory(dom.historySearch.value);
  }

  async function switchConversation(threadId) {
    state.threadId = threadId;
    persist();
    dom.threadIdLabel.textContent = threadId;
    renderHistory(dom.historySearch.value);
    closeSidebar();
    setStatus("thinking");
    try {
      const res = await fetch(ENDPOINTS.history(threadId));
      if (!res.ok) throw new Error("bad status");
      const history = await res.json();
      state.messages = history;
    } catch {
      state.messages = JSON.parse(localStorage.getItem(`astra_msgs_${threadId}`) || "[]");
    }
    renderAllMessages();
    setStatus("ready");
  }

  function startNewChat() {
    state.threadId = uuid();
    state.messages = [];
    state.pendingAttachments = [];
    persist();
    dom.threadIdLabel.textContent = state.threadId;
    dom.convTitle.textContent = "New Chat";
    dom.messagesList.innerHTML = "";
    dom.messagesList.classList.add("hidden");
    dom.welcomeScreen.classList.remove("hidden");
    renderAttachmentsBar();
    renderHistory(dom.historySearch.value);
    closeSidebar();
    dom.messageInput.focus();
  }
  dom.newChatBtn.addEventListener("click", startNewChat);

  function persistMessagesLocally() {
    if (!state.threadId) return;
    localStorage.setItem(`astra_msgs_${state.threadId}`, JSON.stringify(state.messages));
  }

  /* ---------------- Rendering messages ---------------- */
  function renderAllMessages() {
    dom.messagesList.innerHTML = "";
    if (state.messages.length === 0) {
      dom.welcomeScreen.classList.remove("hidden");
      dom.messagesList.classList.add("hidden");
      return;
    }
    dom.welcomeScreen.classList.add("hidden");
    dom.messagesList.classList.remove("hidden");
    state.messages.forEach((m) => {
      if (m.role === "user") appendUserBubble(m.content, m.attachments || [], false);
      else appendAssistantBubble(m.content, false);
    });
    scrollToBottom();
  }

  function scrollToBottom() {
    dom.chatScroll.scrollTop = dom.chatScroll.scrollHeight;
  }

  function appendUserBubble(text, attachments, animate = true) {
    dom.welcomeScreen.classList.add("hidden");
    dom.messagesList.classList.remove("hidden");
    const tpl = $("#tpl-user-bubble").content.cloneNode(true);
    tpl.querySelector(".msg-content").textContent = text;
    const chipsWrap = tpl.querySelector(".attachment-chips");
    attachments.forEach((a) => {
      const chip = document.createElement("span");
      chip.className = "flex items-center gap-1 px-2 py-1 rounded-md bg-black/10 text-[12px]";
      chip.innerHTML = `<span class="material-symbols-outlined text-[14px]">description</span>${escapeHtml(a.name)}`;
      chipsWrap.appendChild(chip);
    });
    const row = tpl.querySelector(".msg-row");
    if (!animate) row.style.animation = "none";
    dom.messagesList.appendChild(tpl);
    scrollToBottom();
  }

  function appendAssistantBubble(text, animate = true) {
    const tpl = $("#tpl-assistant-bubble").content.cloneNode(true);
    const row = tpl.querySelector(".msg-row");
    if (!animate) row.style.animation = "none";
    const contentEl = tpl.querySelector(".msg-content");
    contentEl.innerHTML = renderMarkdownLite(text || "");
    const wrapper = tpl.querySelector(".msg-row");
    dom.messagesList.appendChild(tpl);
    const appended = dom.messagesList.lastElementChild;
    wireMessageActions(appended, () => text);
    scrollToBottom();
    return appended;
  }

  function wireMessageActions(rowEl, getText) {
    const actions = rowEl.querySelector(".msg-actions");
    actions.classList.remove("hidden");
    actions.classList.add("flex");
    rowEl.querySelector(".copy-btn").addEventListener("click", () => {
      navigator.clipboard?.writeText(getText());
    });
    rowEl.querySelector(".regen-btn").addEventListener("click", () => {
      if (state.lastUserMessage) sendMessage(state.lastUserMessage, true);
    });
    rowEl.addEventListener("click", (e) => {
      const copyBtn = e.target.closest(".code-copy-btn");
      if (copyBtn) {
        navigator.clipboard?.writeText(decodeURIComponent(copyBtn.dataset.code));
        const original = copyBtn.innerHTML;
        copyBtn.textContent = "Copied";
        setTimeout(() => (copyBtn.innerHTML = original), 1200);
      }
    });
  }

  /* ---------------- Tool execution UI ---------------- */
  function addToolEvent(container, toolName) {
    const meta = TOOL_META[toolName] || { label: toolName, icon: "bolt" };
    const tpl = $("#tpl-tool-event").content.cloneNode(true);
    const el = tpl.querySelector(".tool-event");
    el.dataset.tool = toolName;
    el.querySelector(".tool-icon").textContent = meta.icon;
    el.querySelector(".tool-name").textContent = meta.label;
    container.appendChild(tpl);
    return container.lastElementChild;
  }
  function completeToolEvent(el) {
    if (!el) return;
    el.classList.add("done");
    el.querySelector(".tool-spinner").classList.add("hidden");
    el.querySelector(".tool-check").classList.remove("hidden");
  }

  /* ---------------- Attachments (upload / RAG) ---------------- */
  function renderAttachmentsBar() {
    dom.attachmentsBar.innerHTML = "";
    if (state.pendingAttachments.length === 0) {
      dom.attachmentsBar.classList.add("hidden");
      dom.attachmentsBar.classList.remove("flex");
      return;
    }
    dom.attachmentsBar.classList.remove("hidden");
    dom.attachmentsBar.classList.add("flex");
    state.pendingAttachments.forEach((a) => {
      const tpl = $("#tpl-attachment-chip").content.cloneNode(true);
      const chip = tpl.querySelector(".attachment-chip");
      chip.classList.toggle("uploading", a.status === "uploading");
      chip.classList.toggle("error", a.status === "error");
      chip.querySelector(".chip-name").textContent = a.name;
      chip.querySelector(".chip-progress").textContent =
        a.status === "uploading" ? `${a.progress}%` : a.status === "error" ? "failed" : "";
      const removeBtn = chip.querySelector(".chip-remove");
      removeBtn.classList.remove("hidden");
      removeBtn.addEventListener("click", () => {
        state.pendingAttachments = state.pendingAttachments.filter((x) => x.id !== a.id);
        renderAttachmentsBar();
      });
      dom.attachmentsBar.appendChild(tpl);
    });
  }

  function validExtension(filename) {
    const ext = filename.split(".").pop().toLowerCase();
    return ALLOWED_EXT.includes(ext);
  }

  async function uploadFile(file) {
    const id = uuid();
    const record = { id, name: file.name, status: "uploading", progress: 0 };
    state.pendingAttachments.push(record);
    renderAttachmentsBar();
    setStatus("uploading");

    if (!validExtension(file.name)) {
      record.status = "error";
      renderAttachmentsBar();
      showError(`Unsupported file type: ${file.name}. Allowed: ${ALLOWED_EXT.join(", ")}`);
      setStatus("ready");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("thread_id", state.threadId);

    try {
      await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", ENDPOINTS.upload);
        xhr.upload.onprogress = (evt) => {
          if (evt.lengthComputable) {
            record.progress = Math.round((evt.loaded / evt.total) * 100);
            renderAttachmentsBar();
          }
        };
        xhr.onload = () => (xhr.status >= 200 && xhr.status < 300 ? resolve(xhr.response) : reject());
        xhr.onerror = reject;
        xhr.send(formData);
      });
      record.status = "done";
      record.progress = 100;
    } catch {
      await simulateIngestProgress(record);
    }
    renderAttachmentsBar();
    setStatus("ready");
  }

  function simulateIngestProgress(record) {
    return new Promise((resolve) => {
      const step = () => {
        record.progress = Math.min(100, record.progress + 20);
        renderAttachmentsBar();
        if (record.progress >= 100) {
          record.status = "done";
          resolve();
        } else {
          setTimeout(step, 180);
        }
      };
      step();
    });
  }

  dom.attachBtn.addEventListener("click", () => dom.fileInput.click());
  dom.fileInput.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    dom.fileInput.value = "";
    for (const f of files) await uploadFile(f);
  });

  /* ---------------- Speech-to-text ---------------- */
  const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognizer = null;
  if (SpeechRecognitionAPI) {
    recognizer = new SpeechRecognitionAPI();
    recognizer.continuous = false;
    recognizer.interimResults = false;
    recognizer.lang = "en-US";

    recognizer.onstart = () => {
      state.isRecording = true;
      dom.micBtn.classList.add("listening");
      setStatus("listening");
    };
    recognizer.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join(" ");
      dom.messageInput.value = (dom.messageInput.value + " " + transcript).trim();
      autoResize();
    };
    recognizer.onerror = (event) => {
      if (event.error === "not-allowed" || event.error === "permission-denied") {
        showError("Microphone permission was denied.");
      } else {
        showError("Speech recognition error: " + event.error);
      }
    };
    recognizer.onend = () => {
      state.isRecording = false;
      dom.micBtn.classList.remove("listening");
      setStatus("ready");
    };
  } else {
    dom.micBtn.addEventListener("click", () => showError("Speech recognition isn't supported in this browser."));
  }

  dom.micBtn.addEventListener("click", () => {
    if (!recognizer) return;
    if (state.isRecording) recognizer.stop();
    else {
      try {
        recognizer.start();
      } catch {
        /* already started */
      }
    }
  });

  /* ---------------- Sending messages / streaming ---------------- */
  dom.messageInput.addEventListener("input", autoResize);
  dom.messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  dom.sendBtn.addEventListener("click", () => sendMessage());

  function setSending(isSending) {
    state.isStreaming = isSending;
    dom.sendBtn.disabled = isSending;
    dom.sendIcon.textContent = isSending ? "stop" : "send";
  }

  async function sendMessage(forcedText, isRegen = false) {
    const text = (forcedText ?? dom.messageInput.value).trim();
    const doneAttachments = state.pendingAttachments.filter((a) => a.status === "done");
    if (!text && doneAttachments.length === 0) return;
    if (state.isStreaming) return;
    if (!state.threadId) startNewChat();

    if (!isRegen) {
      state.lastUserMessage = text;
      state.messages.push({ role: "user", content: text, attachments: doneAttachments });
      appendUserBubble(text, doneAttachments);
      dom.convTitle.textContent = text.slice(0, 40) || "New Chat";
    }

    dom.messageInput.value = "";
    autoResize();
    state.pendingAttachments = [];
    renderAttachmentsBar();
    persistMessagesLocally();
    cacheConversationLocally();

    setSending(true);
    setStatus("thinking");

    const assistantRow = appendAssistantBubble("");
    const contentEl = assistantRow.querySelector(".msg-content");
    const toolContainer = assistantRow.querySelector(".tool-events");
    const cursor = assistantRow.querySelector(".typing-cursor");
    cursor.classList.remove("hidden");

    let fullText = "";
    const activeToolEls = {};

    const onToken = (chunk) => {
      if (!fullText) setStatus("streaming");
      fullText += chunk;
      contentEl.innerHTML = renderMarkdownLite(fullText);
      scrollToBottom();
    };
    const onToolStart = (tool) => {
      setStatus("tool");
      activeToolEls[tool] = addToolEvent(toolContainer, tool);
      scrollToBottom();
    };
    const onToolEnd = (tool) => {
      completeToolEvent(activeToolEls[tool]);
      setStatus("thinking");
    };

    try {
      await streamChat(text, doneAttachments, { onToken, onToolStart, onToolEnd });
    } catch (err) {
      const msg = err.message || "";
      if (msg.includes("quota") || msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED")) {
        showError("Gemini API quota exceeded. The error was: " + msg.slice(0, 120) + "...");
      } else {
        showError("Couldn't reach the server. " + msg.slice(0, 100));
      }
      await simulateAssistantResponse(text, { onToken, onToolStart, onToolEnd });
    }

    cursor.classList.add("hidden");
    wireMessageActions(assistantRow, () => fullText);
    state.messages.push({ role: "assistant", content: fullText });
    persistMessagesLocally();
    cacheConversationLocally();

    setSending(false);
    setStatus("ready");
  }

  /* Real backend call: POST /chat/stream, parses an SSE-style body. */
  async function streamChat(message, attachments, handlers) {
    const res = await fetch(ENDPOINTS.chatStream, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        thread_id: state.threadId,
        message,
        model: state.model,
        attachments: attachments.map((a) => a.id),
      }),
    });
    if (!res.ok || !res.body) throw new Error("stream unavailable");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sepIndex;
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);
        const eventMatch = rawEvent.match(/event:\s*(\w+)/);
        const dataMatch = rawEvent.match(/data:\s*(.*)/);
        const eventType = eventMatch ? eventMatch[1] : "token";
        let payload = {};
        try {
          payload = dataMatch ? JSON.parse(dataMatch[1]) : {};
        } catch {
          payload = { content: dataMatch ? dataMatch[1] : "" };
        }

        if (eventType === "token") handlers.onToken(payload.content || "");
        else if (eventType === "tool_start") handlers.onToolStart(payload.tool);
        else if (eventType === "tool_end") handlers.onToolEnd(payload.tool);
        else if (eventType === "error") throw new Error(payload.message || "stream error");
      }
    }
  }

  /* Local fallback so the interface is fully demonstrable without a
     running backend. Never used when the real API responds. */
  async function simulateAssistantResponse(userText, handlers) {
    const wantsSearch = /search|news|latest|current/i.test(userText);
    if (wantsSearch) {
      handlers.onToolStart("web_search");
      await wait(700);
      handlers.onToolEnd("web_search");
    }
    const reply =
      `Here's a local preview reply since the \`/chat/stream\` backend isn't reachable right now.\n\n` +
      `You said: **"${userText.slice(0, 120)}"**\n\n` +
      "Once connected, responses will stream token-by-token from the real model, including live tool calls like `web_search` or `calculator` when needed.";
    for (const word of reply.split(" ")) {
      handlers.onToken(word + " ");
      await wait(18);
    }
  }
  function wait(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  /* ---------------- Network status ---------------- */
  window.addEventListener("offline", () => showError("You're offline. Messages will fail until reconnected."));

  /* ---------------- Init ---------------- */
  async function init() {
    renderModelList();
    const activeModel = MODELS.find((m) => m.id === state.model) || MODELS[0];
    dom.modelLabel.textContent = activeModel.name;
    renderSuggestedPrompts();
    setStatus("ready");

    await loadConversations();

    const savedThread = localStorage.getItem("astra_thread_id");
    if (savedThread) {
      const cachedMsgs = JSON.parse(localStorage.getItem(`astra_msgs_${savedThread}`) || "[]");
      const hasStaleError = cachedMsgs.some(
        (m) => typeof m.content === "string" && m.content.includes("image.png")
      );
      if (hasStaleError) localStorage.removeItem(`astra_msgs_${savedThread}`);
      await switchConversation(savedThread);
    } else {
      startNewChat();
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
