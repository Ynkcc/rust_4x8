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

let selectedSquare = null; // 记录当前选中的格子索引 (0-31 for 4x8)
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
  // 确保样式正确
  boardEl.style.display = 'grid';
  boardEl.style.gridTemplateColumns = 'repeat(8, 1fr)';
  boardEl.style.gridTemplateRows = 'repeat(4, 1fr)';
  boardEl.style.gap = '5px';

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
  const btn = document.getElementById('btn-new-game');
  if (btn) {
    btn.onclick = async () => {
      console.log("Starting new game...");
      selectedSquare = null;
      try {
        // 读取对手设置并传递给后端
        const oppSel = document.getElementById('opponent-select');
        const opponent = oppSel ? oppSel.value : 'PvP';
        const state = await invoke("reset_game", { opponent });
        await updateUI(state);
      } catch (e) {
        console.error("Reset game failed:", e);
        alert("重置游戏失败: " + e);
      }
    };
  }

  // 绑定 MCTS+DL 控件 (暂时禁用)
  const refreshBtn = document.getElementById('btn-refresh-models');
  const loadBtn = document.getElementById('btn-load-model');
  const modelSelect = document.getElementById('model-select');
  const applyItersBtn = document.getElementById('btn-apply-iters');
  const itersInput = document.getElementById('mcts-iters');

  async function refreshModels() {
    // 模型功能暂时禁用
    console.log('模型功能暂时禁用');
    if (modelSelect) {
      modelSelect.innerHTML = '<option value="">模型功能暂时禁用</option>';
    }
  }

  if (refreshBtn) refreshBtn.onclick = refreshModels;
  if (loadBtn) loadBtn.onclick = async () => {
    alert('模型功能暂时禁用');
  };
  if (applyItersBtn) applyItersBtn.onclick = async () => {
    alert('模型功能暂时禁用');
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