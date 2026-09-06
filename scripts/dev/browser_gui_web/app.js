const params = new URLSearchParams(window.location.search);
const token = params.get("token") || "";
const state = { snapshot: null, ui: [], logs: [], artifacts: [] };

const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-PlaScan-Debug-Token": token,
      ...(options.headers || {}),
    },
  });
  const body = await response.json();
  if (!response.ok || !body.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body.result ?? body;
}

function toast(message, failed = false) {
  const node = byId("toast");
  node.textContent = message;
  node.classList.toggle("failed", failed);
  node.classList.remove("hidden");
  window.setTimeout(() => node.classList.add("hidden"), 3200);
}

function emptyState(icon, title, detail) {
  return `<span class="empty-icon">${escapeHtml(icon)}</span><strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small>`;
}

function formatTime(timestamp) {
  if (!timestamp) return "—";
  const parsed = new Date(timestamp);
  return Number.isNaN(parsed.valueOf()) ? timestamp : parsed.toLocaleTimeString("zh-CN", { hour12: false });
}

function renderSummary(snapshot) {
  state.snapshot = snapshot;
  const project = snapshot.project || {};
  const tasks = snapshot.tasks || [];
  const application = snapshot.application || {};
  const projectName = project.open ? (project.path.split(/[\\/]/).pop() || project.path) : "未打开工程";
  byId("projectState").textContent = projectName;
  byId("projectPath").textContent = project.path || "启动后可通过命令行加载隔离工程副本";
  byId("projectPath").title = project.path || "";
  byId("imageCount").textContent = project.image_count ?? 0;
  byId("taskCount").textContent = tasks.length;
  byId("activeWindow").textContent = application.active_window || "—";
  byId("modalWindow").textContent = application.modal_window || "无";
  byId("projectDirty").textContent = project.open ? (project.dirty ? "有未保存改动" : "已保存") : "未打开";
  byId("snapshotTime").textContent = formatTime(snapshot.timestamp);
  byId("lastUpdated").textContent = `快照 ${formatTime(snapshot.timestamp)}`;

  const taskList = byId("tasks");
  taskList.classList.toggle("empty-state", tasks.length === 0);
  taskList.innerHTML = tasks.length ? tasks.map((task) => {
    const maximum = Number(task.progress_maximum || 0);
    const value = Number(task.progress_value || 0);
    const ratio = maximum > 0 ? Math.max(0, Math.min(100, value * 100 / maximum)) : 0;
    const detail = task.detail_text || (maximum > 0 ? `${value}/${maximum}` : "运行中");
    return `<article class="record"><div class="record-head"><strong>${escapeHtml(task.status_text || task.object_name)}</strong><code>${escapeHtml(task.object_name)}</code></div><p>${escapeHtml(detail)}</p><progress class="task-progress" max="100" value="${ratio}">${ratio}%</progress></article>`;
  }).join("") : emptyState("✓", "当前没有活动任务", "任务启动后，阶段、进度和取消状态会出现在这里。");

  const recent = snapshot.recent_error || "";
  byId("recentError").textContent = recent;
  byId("recentError").classList.toggle("hidden", !recent);
  if (Array.isArray(project.artifacts)) {
    renderArtifacts(project.artifacts);
  } else if (snapshot.counts) {
    byId("artifactCount").textContent = snapshot.counts.artifacts ?? state.artifacts.length;
  }
}

function flattenUi(values, output = []) {
  for (const value of values || []) {
    if (value.object_name) output.push(value);
    flattenUi(value.children || [], output);
    for (const action of value.actions || []) if (action.object_name) output.push(action);
  }
  return output;
}

function renderUi() {
  const filter = byId("uiFilter").value.trim().toLowerCase();
  const allObjects = flattenUi(state.ui);
  const objects = allObjects.filter((item) => {
    const haystack = `${item.object_name} ${item.text || ""} ${item.window_title || ""} ${item.class || ""}`.toLowerCase();
    return !filter || haystack.includes(filter);
  }).slice(0, 400);
  byId("uiCount").textContent = allObjects.length || "0";
  const container = byId("uiObjects");
  container.classList.toggle("empty-state", objects.length === 0);
  container.innerHTML = objects.length ? objects.map((item) => {
    const activatable = item.enabled && (String(item.class).includes("Button") || String(item.class).includes("Action"));
    const status = `${item.class} · ${item.visible ? "可见" : "隐藏"} · ${item.enabled ? "可用" : "禁用"}`;
    return `<article class="record"><div class="record-head"><div><strong>${escapeHtml(item.object_name)}</strong><p>${escapeHtml(status)}</p></div>${activatable ? `<button class="button secondary ui-activate" data-object-name="${escapeHtml(item.object_name)}">触发</button>` : ""}</div><p>${escapeHtml(item.text || item.window_title || item.current_text || "")}</p></article>`;
  }).join("") : emptyState("⌕", "没有匹配的具名控件", "尝试缩短关键词，或重新读取最新控件树。");
  document.querySelectorAll(".ui-activate").forEach((button) => {
    button.addEventListener("click", () => interact(button.dataset.objectName, "activate"));
  });
}

function renderLogs() {
  const level = byId("logLevel").value;
  const logs = state.logs.filter((entry) => level === "all" || entry.level === level);
  byId("logCount").textContent = state.logs.length;
  byId("logs").textContent = logs.length
    ? logs.map((entry) => entry.formatted || `[${entry.level}] ${entry.message}`).join("")
    : "暂无符合当前筛选条件的日志";
  byId("logs").scrollTop = byId("logs").scrollHeight;
}

function renderArtifacts(artifacts) {
  state.artifacts = artifacts;
  byId("artifactCount").textContent = artifacts.length;
  const container = byId("artifacts");
  container.classList.toggle("empty-state", artifacts.length === 0);
  container.innerHTML = artifacts.length ? artifacts.map((artifact) => `<article class="record"><div class="record-head"><strong>${escapeHtml(artifact.key)}</strong><code>${artifact.exists ? "存在" : "缺失"}</code></div><p class="mono">${escapeHtml(artifact.path)}</p></article>`).join("") : emptyState("◇", "当前没有登记产物", "工程报告、点云、网格与地形成果会列在这里。");
}

async function interact(objectName, operation, value) {
  try {
    await api("/api/interact", { method: "POST", body: JSON.stringify({ object_name: objectName, operation, value }) });
    toast(`已调度 ${operation}: ${objectName}`);
    window.setTimeout(refreshSnapshot, 300);
  } catch (error) { toast(error.message, true); }
}

async function refreshSnapshot() {
  try { renderSummary(await api("/api/snapshot")); setHealth(true); }
  catch (error) { setHealth(false, error.message); }
}

async function refreshUi() {
  try { state.ui = await api("/api/ui-tree"); renderUi(); }
  catch (error) { toast(error.message, true); }
}

async function refreshLogs() {
  try { state.logs = await api("/api/logs"); renderLogs(); }
  catch (error) { toast(error.message, true); }
}

async function refreshAll() {
  const activeTab = document.querySelector(".tab.active")?.dataset.tab;
  const requests = [refreshSnapshot(), refreshLogs()];
  if (activeTab === "ui") requests.push(refreshUi());
  await Promise.all(requests);
  toast("调试状态已刷新");
}

function setHealth(ok, detail = "") {
  const node = byId("bridgeHealth");
  node.innerHTML = `<i></i>${ok ? "调试桥在线" : "调试桥异常"}`;
  node.title = detail;
  node.className = `health-pill ${ok ? "ready" : "error"}`;
}

document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((item) => {
    const active = item === button;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
  byId(`${button.dataset.tab}Panel`).classList.add("active");
}));
byId("globalRefresh").addEventListener("click", refreshAll);
byId("refreshSnapshot").addEventListener("click", refreshSnapshot);
byId("refreshUi").addEventListener("click", refreshUi);
byId("uiFilter").addEventListener("input", renderUi);
byId("refreshLogs").addEventListener("click", refreshLogs);
byId("logLevel").addEventListener("change", renderLogs);
byId("closeDialog").addEventListener("click", async () => {
  try {
    await api("/api/close-dialog", { method: "POST", body: "{}" });
    toast("已请求关闭模态框");
    window.setTimeout(refreshSnapshot, 300);
  } catch (error) { toast(error.message, true); }
});
byId("captureScreenshot").addEventListener("click", async () => {
  try {
    const capture = await api("/api/screenshot");
    if (!capture.available) throw new Error("当前窗口无法捕获");
    byId("debugScreenshot").src = `data:image/png;base64,${capture.data_base64}`;
    byId("debugScreenshot").classList.remove("hidden");
    toast("已捕获当前 Qt 窗口");
  } catch (error) { toast(error.message, true); }
});

(async () => {
  if (!token) { setHealth(false, "URL 缺少会话令牌"); return; }
  try {
    const session = await api("/api/session");
    byId("runId").textContent = session.run_id;
    byId("vncFrame").src = session.novnc_url;
    byId("openVnc").href = session.novnc_url;
    await Promise.all([refreshSnapshot(), refreshLogs()]);
    const events = new EventSource(`/api/events?token=${encodeURIComponent(token)}`);
    events.addEventListener("status", (event) => {
      const message = JSON.parse(event.data);
      renderSummary(message.data || message);
      setHealth(true);
    });
    events.onerror = () => setHealth(false, "状态事件流断开，浏览器会自动重连");
  } catch (error) { setHealth(false, error.message); }
})();
