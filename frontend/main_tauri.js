// Tauri API 访问 - 兼容 Tauri v2
let invoke;
if (window.__TAURI__) {
  // Tauri v1 风格
  invoke = window.__TAURI__.core?.invoke || window.__TAURI__.invoke;
} else if (window.__TAURI_INTERNALS__) {
  // Tauri v2 风格
  invoke = window.__TAURI_INTERNALS__.invoke;
} else {
  console.error("Tauri API not found!");
  invoke = async () => { throw new Error("Tauri API not available"); };
}

let selectedSquare = null; // 记录当前选中的格子索引（按变体：4x8=32 格 / 4x4=16 格 / mini=8 格）
let gameState = null;

const pieceTypeOrder = [
  "General",
  "Advisor",
  "Elephant",
  "Chariot",
  "Horse",
  "Cannon",
  "Soldier",
];

const pieceTypeMeta = {
  General: { code: "Gen", redChar: "帥", blackChar: "將" },
  Advisor: { code: "Adv", redChar: "仕", blackChar: "士" },
  Elephant: { code: "Ele", redChar: "相", blackChar: "象" },
  Chariot: { code: "Car", redChar: "車", blackChar: "車" },
  Horse: { code: "Hor", redChar: "馬", blackChar: "馬" },
  Cannon: { code: "Can", redChar: "炮", blackChar: "砲" },
  Soldier: { code: "Sol", redChar: "兵", blackChar: "卒" },
};

const actionIndexCache = new Map();
let moveHighlightMap = new Map();
let revealHighlightSet = new Set();

const bitboardOrder = [
  { key: 'hidden', label: 'Hidden (暗子)' },
  { key: 'empty', label: 'Empty (空位)' },
  { key: 'red_revealed', label: 'Red All (红方)' },
  { key: 'black_revealed', label: 'Black All (黑方)' },
  { key: 'red_soldier', label: 'R_Sol (红兵)' },
  { key: 'black_soldier', label: 'B_Sol (黑卒)' },
  { key: 'red_advisor', label: 'R_Adv (红仕)' },
  { key: 'black_advisor', label: 'B_Adv (黑士)' },
  { key: 'red_general', label: 'R_Gen (红帅)' },
  { key: 'black_general', label: 'B_Gen (黑将)' },
  { key: 'red_cannon', label: 'R_Can (红炮)' },
  { key: 'black_cannon', label: 'B_Can (黑砲)' },
  { key: 'red_horse', label: 'R_Hor (红馬)' },
  { key: 'black_horse', label: 'B_Hor (黑馬)' },
  { key: 'red_chariot', label: 'R_Car (红車)' },
  { key: 'black_chariot', label: 'B_Car (黑車)' },
  { key: 'red_elephant', label: 'R_Ele (红相)' },
  { key: 'black_elephant', label: 'B_Ele (黑象)' },
];

// 棋子文字映射
function getPieceText(slotStr) {
  if (slotStr === "Empty") return "";
  if (slotStr === "Hidden") return "?";
  
  // 格式: "R_Sol", "B_Gen", "R_Adv" 等
  const isRed = slotStr.startsWith("R_");
  const type = slotStr.substring(2); // "Sol", "Gen", "Adv", "Can", "Hor", "Car", "Ele"
  
  if (isRed) {
    if (type === "Gen") return "帥";
    if (type === "Adv") return "仕";
    if (type === "Sol") return "兵";
    if (type === "Can") return "炮";
    if (type === "Hor") return "馬";
    if (type === "Car") return "車";
    if (type === "Ele") return "相";
  } else {
    if (type === "Gen") return "將";
    if (type === "Adv") return "士";
    if (type === "Sol") return "卒";
    if (type === "Can") return "砲";
    if (type === "Hor") return "馬";
    if (type === "Car") return "車";
    if (type === "Ele") return "象";
  }
  return "";
}

// 判断是否是明子
function isRevealed(slotStr) {
  return slotStr !== "Empty" && slotStr !== "Hidden";
}

// 获取玩家
function getSlotPlayer(slotStr) {
  if (slotStr.startsWith("R_")) return "Red";
  if (slotStr.startsWith("B_")) return "Black";
  return null;
}

function isSelectablePiece(state, idx) {
  if (!state || idx == null) return false;
  const slot = state.board?.[idx];
  if (!isRevealed(slot)) return false;
  return getSlotPlayer(slot) === state.current_player;
}

function updateCurrentPlayerIndicator(player) {
  const indicator = document.getElementById('current-player');
  if (!indicator) return;
  const textEl = indicator.querySelector('.indicator-text');
  indicator.classList.toggle('black-turn', player === 'Black');
  if (textEl) {
    textEl.textContent = player === 'Red' ? '红方' : '黑方';
  }
}

function renderFallenPieces(player, deadList) {
  const targetId = player === 'Red' ? 'dead-red' : 'dead-black';
  const container = document.getElementById(targetId);
  if (!container) return;
  container.innerHTML = '';

  const counts = {};
  pieceTypeOrder.forEach(type => { counts[type] = 0; });
  (deadList || []).forEach(typeName => {
    if (counts.hasOwnProperty(typeName)) {
      counts[typeName] += 1;
    }
  });

  pieceTypeOrder.forEach(typeName => {
    const meta = pieceTypeMeta[typeName];
    if (!meta) return;
    const count = counts[typeName] || 0;
    const item = document.createElement('div');
    item.className = 'fallen-item ' + (count > 0 ? 'has-loss' : 'no-loss');

    const icon = document.createElement('span');
    icon.className = `fallen-icon ${player === 'Red' ? 'red' : 'black'}`;
    icon.textContent = player === 'Red' ? meta.redChar : meta.blackChar;
    if (count > 1) {
      icon.setAttribute('data-count', count);
    }

    const label = document.createElement('span');
    label.className = 'fallen-label';
    label.textContent = player === 'Red' ? meta.redChar : meta.blackChar;

    item.appendChild(icon);
    item.appendChild(label);
    container.appendChild(item);
  });
}

function computeRevealHighlights(state) {
  const revealSet = new Set();
  if (!state || !state.board) return revealSet;
  state.board.forEach((slot, idx) => {
    if (slot === 'Hidden' && state.action_masks?.[idx] === 1) {
      revealSet.add(idx);
    }
  });
  return revealSet;
}

async function computeMoveHighlights(state, fromIdx) {
  const highlights = new Map();
  if (!state || fromIdx == null) return highlights;
  const tasks = state.board.map(async (_slot, idx) => {
    if (idx === fromIdx) return null;
    const action = await getCachedMoveAction(fromIdx, idx);
    if (action == null) return null;
    if (state.action_masks?.[action] !== 1) return null;
    const type = getMoveHighlightType(state.board[idx], state.current_player);
    if (!type) return null;
    return { idx, type };
  });

  const results = await Promise.all(tasks);
  results.filter(Boolean).forEach(entry => {
    highlights.set(entry.idx, entry);
  });
  return highlights;
}

async function getCachedMoveAction(fromSq, toSq) {
  const key = `${fromSq}-${toSq}`;
  if (actionIndexCache.has(key)) {
    return actionIndexCache.get(key);
  }
  try {
    const action = await invoke('get_move_action', { fromSq, toSq });
    actionIndexCache.set(key, action);
    return action;
  } catch (err) {
    console.error('get_move_action failed', err);
    actionIndexCache.set(key, null);
    return null;
  }
}

function getMoveHighlightType(slot, currentPlayer) {
  if (!slot || slot === 'Empty') return 'move';
  if (slot === 'Hidden') return null;
  if (isRevealed(slot)) {
    return getSlotPlayer(slot) !== currentPlayer ? 'capture' : null;
  }
  return null;
}

async function updateUI(state) {
  console.log("updateUI called with state:", state);
  gameState = state;
  
  // 检查 state 是否有效
  if (!state || !state.board) {
    console.error("Invalid state received:", state);
    return;
  }
  // 变体判断：mini = 4x2 迷你暗棋（8 格），4x4 = 4x4 暗棋（16 格），dark = 4x8 暗棋（32 格）
  const isMini = state.variant === 'mini' || state.board.length === 8;
  const is4x4 = state.variant === '4x4' || state.board.length === 16;
  const maxHp = isMini ? 47 : 60; // 4x4 与 4x8 血量上限均为 60
  if (selectedSquare !== null && !isSelectablePiece(state, selectedSquare)) {
    selectedSquare = null;
  }
  
  console.log("Board length:", state.board.length);
  
  // 1. 更新状态栏
  updateCurrentPlayerIndicator(state.current_player);
  
  let statusText = "进行中";
  if (state.winner === 1) statusText = "红方获胜";
  else if (state.winner === -1) statusText = "黑方获胜";
  else if (state.winner === 0 && state.total_step_counter > 0) statusText = "和棋"; // 简单判断，实际应由后端返回 terminated
  
  document.getElementById('game-status').value = statusText;
  document.getElementById('move-counter').value = state.move_counter;
  // 更新血量显示（如果后端返回 hp_red / hp_black ）
  try {
    const hpRedEl = document.getElementById('hp-red');
    const hpBlackEl = document.getElementById('hp-black');
    const hpRedFill = document.getElementById('hp-red-fill');
    const hpBlackFill = document.getElementById('hp-black-fill');
    if (hpRedEl && typeof state.hp_red !== 'undefined') {
      hpRedEl.value = String(state.hp_red);
      if (hpRedFill) {
        const pct = Math.max(0, Math.min(100, Math.round((state.hp_red / maxHp) * 100)));
        hpRedFill.style.width = pct + "%";
      }
    }
    if (hpBlackEl && typeof state.hp_black !== 'undefined') {
      hpBlackEl.value = String(state.hp_black);
      if (hpBlackFill) {
        const pct = Math.max(0, Math.min(100, Math.round((state.hp_black / maxHp) * 100)));
        hpBlackFill.style.width = pct + "%";
      }
    }
  } catch (e) {
    console.warn('HP update skipped, missing elements or state fields', e);
  }

  // 更新阵亡棋子
  renderFallenPieces('Red', state.dead_red);
  renderFallenPieces('Black', state.dead_black);
  
  // 更新隐藏棋子
  renderHiddenPieces('Red', state.hidden_red);
  renderHiddenPieces('Black', state.hidden_black);

  // 计算高亮
  revealHighlightSet = computeRevealHighlights(state);
  if (selectedSquare !== null) {
    moveHighlightMap = await computeMoveHighlights(state, selectedSquare);
  } else {
    moveHighlightMap = new Map();
  }

  // 2. 渲染棋盘 (4行8列)
  const boardEl = document.getElementById('chess-board');
  if (!boardEl) {
    console.error("chess-board element not found!");
    return;
  }
  
  boardEl.innerHTML = '';
  // 确保样式正确（mini = 4x2 迷你棋盘，4x4 = 4x4 暗棋棋盘，dark = 4x8 标准棋盘）
  let boardCols, boardRows;
  if (isMini) {
    boardCols = 2;
    boardRows = 4;
  } else if (is4x4) {
    boardCols = 4;
    boardRows = 4;
  } else {
    boardCols = 8;
    boardRows = 4;
  }
  boardEl.style.display = 'grid';
  boardEl.style.gridTemplateColumns = `repeat(${boardCols}, 1fr)`;
  boardEl.style.gridTemplateRows = `repeat(${boardRows}, 1fr)`;
  boardEl.style.gap = '5px';
  if (isMini) {
    boardEl.style.width = 'min(260px, 60vw)';
    boardEl.style.aspectRatio = '2 / 4';
  } else if (is4x4) {
    boardEl.style.width = 'min(440px, 80vw)';
    boardEl.style.aspectRatio = '4 / 4';
  } else {
    boardEl.style.width = 'min(700px, 90vw)';
    boardEl.style.aspectRatio = '8 / 4';
  }

  console.log("Rendering board with", state.board.length, "cells");

  state.board.forEach((slot, idx) => {
    const cell = document.createElement('div');
    cell.className = 'chess-cell';
    
    // 样式类
    if (slot === "Hidden") {
      cell.classList.add('hidden');
    } else if (slot === "Empty") {
      cell.classList.add('empty');
    } else {
      const player = getSlotPlayer(slot);
      cell.classList.add(player === "Red" ? 'red' : 'black');
    }
    
    if (selectedSquare === idx) {
      cell.classList.add('selected');
    }

    if (revealHighlightSet.has(idx) && slot === 'Hidden') {
      cell.classList.add('legal-reveal');
    }
    const moveHighlight = moveHighlightMap.get(idx);
    if (moveHighlight) {
      cell.classList.add(moveHighlight.type === 'capture' ? 'legal-capture' : 'legal-move');
    }

    const pieceText = getPieceText(slot);
    cell.textContent = pieceText;
    console.log(`Cell ${idx}: ${slot} -> ${pieceText}`);
    cell.onclick = () => onSquareClick(idx);
    boardEl.appendChild(cell);
  });
  
  console.log("Board rendered with", boardEl.children.length, "cells");

  renderBitboards(state.bitboards);
}

async function onSquareClick(idx) {
  if (!gameState) return;

  const slot = gameState.board[idx];
  const actionMasks = gameState.action_masks;

  // 如果当前未选中
  if (selectedSquare === null) {
    if (slot === "Hidden") {
      // 尝试翻开 (action 0-11)
      const action = idx;
      if (actionMasks[action] === 1) {
        try {
          const result = await invoke("step_game", { action });
          await updateUI(result.state);
          checkGameOver(result);
          // 若对手为电脑，则让电脑走一步
          if (!(result.terminated || result.truncated)) {
            await maybeBotTurn();
          }
        } catch (e) {
          alert("操作失败: " + e);
        }
      } else {
        console.log("当前位置不可翻开 (Action Mask Restricted)");
      }
    } else if (isRevealed(slot)) {
      const player = getSlotPlayer(slot);
      if (player === gameState.current_player) {
        selectedSquare = idx;
        await updateUI(gameState);
      }
    }
  } else {
    // 已有选中
    if (idx === selectedSquare) {
      selectedSquare = null; // 取消
      await updateUI(gameState);
    } else if (isRevealed(slot) && getSlotPlayer(slot) === gameState.current_player) {
      selectedSquare = idx; // 切换选中
      await updateUI(gameState);
    } else {
      // 尝试移动
      try {
        const action = await invoke("get_move_action", { 
          fromSq: selectedSquare, 
          toSq: idx 
        });
        
        if (action !== null && actionMasks[action] === 1) {
          const result = await invoke("step_game", { action });
          selectedSquare = null;
          await updateUI(result.state);
          checkGameOver(result);
          // 若对手为电脑，则让电脑走一步
          if (!(result.terminated || result.truncated)) {
            await maybeBotTurn();
          }
        } else {
          console.log("无效移动 (Action Mask Restricted)");
          // 也可以选择 selectedSquare = null; 取消选中
        }
      } catch (e) {
        console.error("Move calculation error:", e);
      }
    }
  }
}

// 根据当前选择的对手模式，只显示对应的 AI 设置面板
function updateAiSettingsVisibility() {
  const oppSel = document.getElementById('opponent-select');
  if (!oppSel) return;
  const opponent = oppSel.value;

  const settingsMap = {
    Minimax: 'settings-minimax',
    Engine: 'settings-engine',
    MctsHeuristic: 'settings-heuristic',
    MctsDL: 'settings-mctsdl',
    MctsOnnx: 'settings-mctsdl',
  };
  const targetId = settingsMap[opponent];

  ['settings-minimax', 'settings-engine', 'settings-heuristic', 'settings-mctsdl'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.style.display = (id === targetId) ? '' : 'none';
    }
  });
}

function checkGameOver(result) {
  if (result.terminated || result.truncated) {
    setTimeout(() => {
       let msg = "游戏结束！";
       if (result.winner === 1) msg += " 红方获胜！";
       else if (result.winner === -1) msg += " 黑方获胜！";
       else if (result.winner === 0) msg += " 平局！";
       alert(msg);
    }, 100);
  }
}

// 启动
window.addEventListener('DOMContentLoaded', async () => {
  const oppSel = document.getElementById('opponent-select');
  if (oppSel) {
    oppSel.addEventListener('change', updateAiSettingsVisibility);
  }
  updateAiSettingsVisibility();

  const btn = document.getElementById('btn-new-game');
  if (btn) {
    btn.onclick = async () => {
      console.log("Starting new game...");
      selectedSquare = null;
      try {
        // 读取对手与变体设置并传递给后端
        const oppSel = document.getElementById('opponent-select');
        const opponent = oppSel ? oppSel.value : 'PvP';
        const variantSel = document.getElementById('variant-select');
        const variant = variantSel ? variantSel.value : 'dark';
        const state = await invoke("reset_game", { opponent, variant });
        await updateUI(state);
      } catch (e) {
        console.error("Reset game failed:", e);
        alert("重置游戏失败: " + e);
      }
    };
  }

  // 绑定 MCTS+DL 控件（模型加载 / 列表 / 搜索次数）
  const refreshBtn = document.getElementById('btn-refresh-models');
  const loadBtn = document.getElementById('btn-load-model');
  const modelSelect = document.getElementById('model-select');
  const applyItersBtn = document.getElementById('btn-apply-iters');
  const itersInput = document.getElementById('mcts-iters');
  const modelPathInput = document.getElementById('model-path-input');

  async function refreshModels() {
    if (!modelSelect) return;
    try {
      const models = await invoke('list_models');
      modelSelect.innerHTML = '';
      if (!models || models.length === 0) {
        modelSelect.innerHTML = '<option value="">未找到 .pt / .onnx 模型</option>';
        return;
      }
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.path;
        opt.textContent = m.name;
        modelSelect.appendChild(opt);
      });
    } catch (e) {
      console.error('list_models failed:', e);
      modelSelect.innerHTML = '<option value="">加载模型列表失败</option>';
    }
  }

  // 手动输入模型路径（用于 list_models 未覆盖的路径，如 python/game_4x4/ 子目录）
  if (modelPathInput) {
    modelPathInput.addEventListener('input', () => {
      if (modelPathInput.value.trim()) {
        // 输入框非空时以输入路径优先
        modelPathInput.dataset.manual = '1';
      } else {
        delete modelPathInput.dataset.manual;
      }
    });
  }

  function resolveModelPath() {
    // 手动输入的路径优先；否则用下拉框选中的路径
    if (modelPathInput && modelPathInput.value.trim()) {
      return modelPathInput.value.trim();
    }
    return modelSelect ? modelSelect.value : '';
  }

  if (refreshBtn) refreshBtn.onclick = refreshModels;
  if (loadBtn) loadBtn.onclick = async () => {
    const path = resolveModelPath();
    if (!path) {
      alert('请先选择模型（或手动输入模型路径）');
      return;
    }
    try {
      const result = await invoke('load_model', { path });
      alert('模型加载成功：' + result);
      // 自动切换对手：.onnx → MCTS+ONNX，否则 → MCTS+DL
      const oppSel = document.getElementById('opponent-select');
      if (oppSel) oppSel.value = path.toLowerCase().endsWith('.onnx') ? 'MctsOnnx' : 'MctsDL';
      updateAiSettingsVisibility();
    } catch (e) {
      alert('模型加载失败：' + e);
    }
  };
  if (applyItersBtn) applyItersBtn.onclick = async () => {
    const iters = itersInput ? parseInt(itersInput.value, 10) : NaN;
    if (!Number.isFinite(iters) || iters < 1) {
      alert('搜索次数必须是大于 0 的整数');
      return;
    }
    try {
      const result = await invoke('set_mcts_iterations', { iters });
      if (itersInput) itersInput.value = String(result);
      alert('MCTS 搜索次数已设置为 ' + result);
    } catch (e) {
      alert('设置失败：' + e);
    }
  };

  // 绑定 Minimax 深度设置
  const applyDepthBtn = document.getElementById('btn-apply-depth');
  const depthInput = document.getElementById('minimax-depth');
  if (applyDepthBtn) applyDepthBtn.onclick = async () => {
    if (!depthInput) return;
    const depth = parseInt(depthInput.value, 10);
    if (!Number.isFinite(depth) || depth < 1) {
      alert('搜索深度必须是大于 0 的整数');
      return;
    }
    try {
      const result = await invoke('set_minimax_depth', { depth });
      depthInput.value = String(result);
      alert(`Minimax 搜索深度已设置为 ${result}`);
    } catch (e) {
      alert('设置失败: ' + e);
    }
  };

  // 绑定强引擎难度设置
  const applyEngineBtn = document.getElementById('btn-apply-engine');
  const engineLevelSel = document.getElementById('engine-level');
  if (applyEngineBtn) applyEngineBtn.onclick = async () => {
    if (!engineLevelSel) return;
    const budget = parseInt(engineLevelSel.value, 10);
    if (!Number.isFinite(budget) || budget < 1) {
      alert('节点预算必须大于 0');
      return;
    }
    try {
      const result = await invoke('set_engine_budget', { budget });
      alert(`强引擎节点预算已设置为 ${result}`);
    } catch (e) {
      alert('设置失败: ' + e);
    }
  };

  // 绑定启发式 MCTS 模拟次数设置
  const applyHeuristicBtn = document.getElementById('btn-apply-heuristic');
  const heuristicSimsInput = document.getElementById('heuristic-sims');
  if (applyHeuristicBtn) applyHeuristicBtn.onclick = async () => {
    if (!heuristicSimsInput) return;
    const sims = parseInt(heuristicSimsInput.value, 10);
    if (!Number.isFinite(sims) || sims < 1) {
      alert('模拟次数必须是大于 0 的整数');
      return;
    }
    try {
      const result = await invoke('set_heuristic_sims', { sims });
      heuristicSimsInput.value = String(result);
      alert(`启发式 MCTS 模拟次数已设置为 ${result}`);
    } catch (e) {
      alert('设置失败: ' + e);
    }
  };

  // 加载初始状态
  console.log("Loading initial state...");
  try {
    const state = await invoke("get_game_state");
    console.log("Initial state loaded:", state);
    await updateUI(state);
  } catch (e) {
    console.error("Failed to load initial state:", e);
    alert("加载初始状态失败: " + e);
  }

  // 初始刷新模型列表（显示禁用状态）
  await refreshModels();

  setupBitboardSidebar();
});

// 在人类完成一步后，若对手为电脑，则自动让电脑走一步
async function maybeBotTurn() {
  try {
    const oppType = await invoke("get_opponent_type");
    if (oppType === 'PvP') return;
    // 触发一次 AI 行动
    const result = await invoke("bot_move");
    await updateUI(result.state);
    checkGameOver(result);
  } catch (e) {
    // 当处于 PvP 或无棋可走时，后端可能返回错误，此处静默或打印日志
    console.log('bot_move skipped or failed:', e);
  }
}

function renderHiddenPieces(player, hiddenList) {
  const targetId = player === 'Red' ? 'hidden-red' : 'hidden-black';
  const container = document.getElementById(targetId);
  if (!container) return;
  container.innerHTML = '';

  const counts = {};
  pieceTypeOrder.forEach(type => { counts[type] = 0; });
  (hiddenList || []).forEach(typeName => {
    if (counts.hasOwnProperty(typeName)) {
      counts[typeName] += 1;
    }
  });

  pieceTypeOrder.forEach(typeName => {
    const meta = pieceTypeMeta[typeName];
    if (!meta) return;
    const count = counts[typeName] || 0;
    const item = document.createElement('div');
    item.className = 'fallen-item ' + (count > 0 ? 'has-loss' : 'no-loss');

    const icon = document.createElement('span');
    icon.className = `fallen-icon ${player === 'Red' ? 'red' : 'black'}`;
    icon.textContent = player === 'Red' ? meta.redChar : meta.blackChar;
    if (count > 1) {
      icon.setAttribute('data-count', count);
    }

    const label = document.createElement('span');
    label.className = 'fallen-label';
    label.textContent = player === 'Red' ? meta.redChar : meta.blackChar;

    item.appendChild(icon);
    item.appendChild(label);
    container.appendChild(item);
  });
}

function renderBitboards(bitboards) {
  const container = document.getElementById('bitboard-container');
  const toggleBtn = document.getElementById('toggle-bitboard');
  if (!container) return;
  container.innerHTML = '';

  if (!bitboards) {
    if (toggleBtn) {
      toggleBtn.disabled = true;
    }
    const empty = document.createElement('div');
    empty.className = 'bitboard-empty';
    empty.textContent = '暂无通道数据';
    container.appendChild(empty);
    return;
  }

  if (toggleBtn) {
    toggleBtn.disabled = false;
  }

  let rendered = 0;
  bitboardOrder.forEach(({ key, label }) => {
    if (!bitboards[key] || !Array.isArray(bitboards[key])) return;
    rendered += 1;
    const wrapper = document.createElement('div');
    wrapper.className = 'bb-wrapper';

    const bbLabel = document.createElement('div');
    bbLabel.className = 'bb-label';
    bbLabel.textContent = label;

    const grid = document.createElement('div');
    grid.className = 'bb-grid';

    // 根据棋盘尺寸动态调整位板网格列数（mini=4x2，4x4=4x4，4x8=8x4）
    const bbLen = bitboards[key].length;
    let bbCols = 8, bbRows = 4;
    if (bbLen === 8) { bbCols = 2; bbRows = 4; }
    else if (bbLen === 16) { bbCols = 4; bbRows = 4; }
    else if (bbLen === 32) { bbCols = 8; bbRows = 4; }
    else if (bbLen % 4 === 0) { bbCols = bbLen / 4; bbRows = 4; }
    grid.style.gridTemplateColumns = `repeat(${bbCols}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${bbRows}, 1fr)`;

    bitboards[key].forEach(isActive => {
      const cell = document.createElement('div');
      cell.className = `bb-cell ${isActive ? 'active' : ''}`;
      grid.appendChild(cell);
    });

    wrapper.appendChild(bbLabel);
    wrapper.appendChild(grid);
    container.appendChild(wrapper);
  });

  if (rendered === 0) {
    const empty = document.createElement('div');
    empty.className = 'bitboard-empty';
    empty.textContent = '暂无可视化数据';
    container.appendChild(empty);
  }
}

function setupBitboardSidebar() {
  const sidebar = document.getElementById('bitboard-sidebar');
  const overlay = document.getElementById('bitboard-overlay');
  const toggleBtn = document.getElementById('toggle-bitboard');
  const closeBtn = document.getElementById('close-bitboard');
  if (!sidebar || !overlay || !toggleBtn || !closeBtn) return;

  const openSidebar = () => {
    sidebar.classList.add('open');
    overlay.classList.add('active');
  };

  const closeSidebar = () => {
    sidebar.classList.remove('open');
    overlay.classList.remove('active');
  };

  toggleBtn.addEventListener('click', () => {
    if (sidebar.classList.contains('open')) {
      closeSidebar();
    } else if (!toggleBtn.disabled) {
      openSidebar();
    }
  });

  closeBtn.addEventListener('click', closeSidebar);
  overlay.addEventListener('click', closeSidebar);
  window.addEventListener('keydown', (evt) => {
    if (evt.key === 'Escape') {
      closeSidebar();
    }
  });
}