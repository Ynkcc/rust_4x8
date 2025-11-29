// code_files/frontend/main_tauri.js
const { invoke } = window.__TAURI__.core;

let selectedSquare = null; // 记录当前选中的格子索引 (0-11)
let gameState = null;
let currentOpponentType = "PvP"; // 记录当前的对手类型
let isAiTurn = false; // 锁，防止 AI 思考时用户点击

// 棋子文字映射
function getPieceText(slotStr) {
  if (slotStr === "Empty") return "";
  if (slotStr === "Hidden") return "?";
  
  // 格式: "R_Sol", "B_Gen", "R_Adv" 等
  const isRed = slotStr.startsWith("R_");
  const type = slotStr.substring(2); // "Sol", "Gen", "Adv"
  
  if (isRed) {
    if (type === "Gen") return "帥";
    if (type === "Adv") return "仕";
    if (type === "Sol") return "兵";
  } else {
    if (type === "Gen") return "將";
    if (type === "Adv") return "士";
    if (type === "Sol") return "卒";
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

async function updateUI(state) {
  gameState = state;
  
  // 1. 更新状态栏
  const cp = state.current_player === "Red" ? "红方" : "黑方";
  document.getElementById('current-player').value = cp;
  
  let statusText = "进行中";
  if (state.winner === 1) statusText = "红方获胜";
  else if (state.winner === -1) statusText = "黑方获胜";
  else if (state.winner === 0 && state.total_step_counter > 0) statusText = "和棋";
  
  document.getElementById('game-status').value = statusText;
  document.getElementById('move-counter').value = state.move_counter;
  document.getElementById('dead-red').textContent = state.dead_red.join(', ') || "无";
  document.getElementById('dead-black').textContent = state.dead_black.join(', ') || "无";

  // 2. 渲染棋盘 (3行4列)
  const boardEl = document.getElementById('chess-board');
  boardEl.innerHTML = '';
  boardEl.style.display = 'grid';
  boardEl.style.gridTemplateColumns = 'repeat(4, 1fr)';
  boardEl.style.gridTemplateRows = 'repeat(3, 1fr)';
  boardEl.style.gap = '5px';

  state.board.forEach((slot, idx) => {
    const cell = document.createElement('div');
    cell.className = 'chess-cell';
    
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

    cell.textContent = getPieceText(slot);
    cell.onclick = () => onSquareClick(idx);
    boardEl.appendChild(cell);
  });

  // 3. 渲染辅助信息
  renderInfoPanel(state);

  // 4. 检查是否需要 AI 走棋
  checkAiTurn(state);
}

function renderInfoPanel(state) {
  const infoContainer = document.getElementById('info-container');
  infoContainer.innerHTML = ''; 
  
  // 翻子概率
  const probSection = document.createElement('div');
  probSection.style.marginBottom = '15px';
  const probTitle = document.createElement('h4');
  probTitle.textContent = '翻开暗子概率';
  probSection.appendChild(probTitle);
  
  const probList = document.createElement('div');
  probList.className = 'prob-list';
  const probLabels = ["红兵", "红仕", "红帅", "黑卒", "黑士", "黑将"];
  state.reveal_probabilities.forEach((prob, idx) => {
    const div = document.createElement('div');
    div.className = 'prob-item';
    if (prob <= 0.0001) div.style.opacity = '0.5';
    div.textContent = `${probLabels[idx]}: ${(prob * 100).toFixed(1)}%`;
    probList.appendChild(div);
  });
  probSection.appendChild(probList);
  infoContainer.appendChild(probSection);

  // Bitboards (仅在必要时显示，省略部分代码以简洁)
  // ... (原有的 Bitboard 代码可以保留)
}

async function checkAiTurn(state) {
    // 如果游戏已结束，或者当前是 PvP 模式，不操作
    if (state.winner !== null || currentOpponentType === "PvP") {
        isAiTurn = false;
        return;
    }

    // 假设玩家固定执红，电脑执黑
    // 如果当前玩家是 Black，且是对战模式，则触发 AI
    if (state.current_player === "Black") {
        isAiTurn = true;
        document.getElementById('current-player').value += " (思考中...)";
        
        // 延迟一点时间，模拟思考，并让 UI 刷新出来
        setTimeout(async () => {
            try {
                const result = await invoke("bot_move");
                await updateUI(result.state);
                checkGameOver(result);
            } catch (e) {
                console.error("AI move failed:", e);
                alert("AI 思考出错: " + e);
            } finally {
                isAiTurn = false;
            }
        }, 600);
    } else {
        isAiTurn = false;
    }
}

async function onSquareClick(idx) {
  if (!gameState) return;
  if (isAiTurn) return; // AI 回合禁止点击
  if (gameState.winner !== null) return; // 游戏结束

  const slot = gameState.board[idx];
  const actionMasks = gameState.action_masks;
  const currentPlayer = gameState.current_player;

  // 如果当前未选中
  if (selectedSquare === null) {
    if (slot === "Hidden") {
      const action = idx;
      if (actionMasks[action] === 1) {
        executeMove(action);
      }
    } else if (isRevealed(slot)) {
      const player = getSlotPlayer(slot);
      if (player === currentPlayer) {
        selectedSquare = idx;
        await updateUI(gameState);
      }
    }
  } else {
    // 已有选中
    if (idx === selectedSquare) {
      selectedSquare = null; 
      await updateUI(gameState);
    } else if (isRevealed(slot) && getSlotPlayer(slot) === currentPlayer) {
      selectedSquare = idx; 
      await updateUI(gameState);
    } else {
      // 尝试移动
      try {
        const action = await invoke("get_move_action", { fromSq: selectedSquare, toSq: idx });
        if (action !== null && actionMasks[action] === 1) {
          selectedSquare = null; // 移动前取消选中，避免 UI 闪烁
          executeMove(action);
        }
      } catch (e) {
        console.error(e);
      }
    }
  }
}

async function executeMove(action) {
    try {
        const result = await invoke("step_game", { action });
        await updateUI(result.state);
        checkGameOver(result);
    } catch (e) {
        alert("操作失败: " + e);
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
  const select = document.getElementById('opponent-select');
  
  if (btn) {
    btn.onclick = async () => {
      selectedSquare = null;
      currentOpponentType = select.value;
      console.log("Starting new game vs", currentOpponentType);
      
      try {
        const state = await invoke("reset_game", { opponent: currentOpponentType });
        await updateUI(state);
      } catch (e) {
        console.error("Reset game failed:", e);
        alert("重置游戏失败: " + e);
      }
    };
  }

  // 加载初始状态 (默认 PvP)
  try {
    // 初始查询一下后端的状态，如果后端重启过，可能需要同步
    const state = await invoke("get_game_state");
    // 也要同步对手类型
    const oppType = await invoke("get_opponent_type");
    currentOpponentType = oppType;
    select.value = oppType; // 同步 UI
    await updateUI(state);
  } catch (e) {
    console.error("Failed to load initial state:", e);
  }
});