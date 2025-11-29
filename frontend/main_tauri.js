const { invoke } = window.__TAURI__.core;

let selectedSquare = null; // 记录当前选中的格子索引 (0-11)
let gameState = null;

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
  document.getElementById('current-player').value = state.current_player === "Red" ? "红方" : "黑方";
  
  let statusText = "进行中";
  if (state.winner === 1) statusText = "红方获胜";
  else if (state.winner === -1) statusText = "黑方获胜";
  else if (state.winner === 0 && state.total_step_counter > 0) statusText = "和棋"; // 简单判断，实际应由后端返回 terminated
  
  document.getElementById('game-status').value = statusText;
  document.getElementById('move-counter').value = state.move_counter;
  document.getElementById('dead-red').textContent = state.dead_red.join(', ') || "无";
  document.getElementById('dead-black').textContent = state.dead_black.join(', ') || "无";

  // 2. 渲染棋盘 (3行4列)
  const boardEl = document.getElementById('chess-board');
  boardEl.innerHTML = '';
  // 确保样式正确
  boardEl.style.display = 'grid';
  boardEl.style.gridTemplateColumns = 'repeat(4, 1fr)';
  boardEl.style.gridTemplateRows = 'repeat(3, 1fr)';
  boardEl.style.gap = '5px';

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

    cell.textContent = getPieceText(slot);
    cell.onclick = () => onSquareClick(idx);
    boardEl.appendChild(cell);
  });

  // 3. 渲染辅助信息 (概率 + Bitboards)
  const infoContainer = document.getElementById('info-container');
  infoContainer.innerHTML = ''; // 清空
  
  // --- 3.1 翻子概率部分 ---
  const probSection = document.createElement('div');
  probSection.style.marginBottom = '15px';
  
  const probTitle = document.createElement('h4');
  probTitle.textContent = '翻开暗子概率 (Reveal Prob)';
  probSection.appendChild(probTitle);
  
  const probList = document.createElement('div');
  probList.className = 'prob-list'; // 使用 Grid 布局
  
  const probLabels = ["红兵", "红仕", "红帅", "黑卒", "黑士", "黑将"];
  state.reveal_probabilities.forEach((prob, idx) => {
    const div = document.createElement('div');
    div.className = 'prob-item';
    // 只有概率 > 0 才高亮显示，避免视觉杂乱
    if (prob <= 0.0001) div.style.opacity = '0.5';
    div.textContent = `${probLabels[idx]}: ${(prob * 100).toFixed(1)}%`;
    probList.appendChild(div);
  });
  probSection.appendChild(probList);
  infoContainer.appendChild(probSection);
  
  // --- 3.2 Bitboards 可视化 (通道状态) ---
  if (state.bitboards) {
    const bbSection = document.createElement('div');
    
    const bbTitle = document.createElement('h4');
    bbTitle.textContent = '通道状态 (Bitboards)';
    bbSection.appendChild(bbTitle);
    
    const bitboardList = document.createElement('div');
    bitboardList.className = 'bitboard-list';
    
    // 定义显示顺序和标签
    const bbOrder = [
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
    ];
    
    bbOrder.forEach(({ key, label }) => {
      if (state.bitboards[key]) {
        const wrapper = document.createElement('div');
        wrapper.className = 'bb-wrapper';
        
        const bbLabel = document.createElement('div');
        bbLabel.className = 'bb-label';
        bbLabel.textContent = label;
        
        const grid = document.createElement('div');
        grid.className = 'bb-grid';
        
        state.bitboards[key].forEach(isActive => {
          const cell = document.createElement('div');
          cell.className = `bb-cell ${isActive ? 'active' : ''}`;
          grid.appendChild(cell);
        });
        
        wrapper.appendChild(bbLabel);
        wrapper.appendChild(grid);
        bitboardList.appendChild(wrapper);
      }
    });
    
    bbSection.appendChild(bitboardList);
    infoContainer.appendChild(bbSection);
  }
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

  // 绑定 MCTS+DL 控件
  const refreshBtn = document.getElementById('btn-refresh-models');
  const loadBtn = document.getElementById('btn-load-model');
  const modelSelect = document.getElementById('model-select');
  const applyItersBtn = document.getElementById('btn-apply-iters');
  const itersInput = document.getElementById('mcts-iters');

  async function refreshModels() {
    try {
      const list = await invoke('list_models');
      if (modelSelect) {
        modelSelect.innerHTML = '';
        list.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.path;
          opt.textContent = m.name;
          modelSelect.appendChild(opt);
        });
      }
    } catch (e) {
      console.error('列出模型失败:', e);
    }
  }

  if (refreshBtn) refreshBtn.onclick = refreshModels;
  if (loadBtn) loadBtn.onclick = async () => {
    try {
      const path = modelSelect ? modelSelect.value : '';
      if (!path) return alert('请选择一个模型文件 (.ot)。');
      const msg = await invoke('load_model', { path });
      alert(msg);
    } catch (e) {
      alert('加载失败: ' + e);
    }
  };
  if (applyItersBtn) applyItersBtn.onclick = async () => {
    try {
      const v = parseInt(itersInput.value || '0', 10);
      const n = await invoke('set_mcts_iterations', { iters: v });
      alert('已设置搜索次数：' + n);
    } catch (e) {
      alert('设置失败: ' + e);
    }
  };

  // 加载初始状态
  try {
    const state = await invoke("get_game_state");
    await updateUI(state);
  } catch (e) {
    console.error("Failed to load initial state:", e);
  }

  // 初始刷新模型列表
  await refreshModels();
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