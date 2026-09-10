import { computed, reactive } from 'vue';
import { api } from '../api/client';
import type { GameState, Opponent, Variant } from '../api/types';
import { appendLog } from './useLogs';
import { useToast } from './useToast';
import { isRevealed, pieceText, slotPlayer } from '../domain/pieces';

const toast = useToast();

interface MoveHighlight {
  idx: number;
  type: 'move' | 'capture';
}

interface GameStore {
  state: GameState | null;
  busy: boolean;
  selectedSquare: number | null;
  moveHighlights: Map<number, MoveHighlight>;
  lastResult: { terminated: boolean; truncated: boolean; winner: number } | null;
}

const store = reactive<GameStore>({
  state: null,
  busy: false,
  selectedSquare: null,
  moveHighlights: new Map(),
  lastResult: null,
});

const actionIndexCache = new Map<string, number | null>();

const statusText = computed(() => {
  const s = store.state;
  if (!s) return '—';
  if (store.lastResult?.terminated || store.lastResult?.truncated) {
    const w = store.lastResult.winner;
    return w === 1 ? '红方获胜' : w === -1 ? '黑方获胜' : '和棋';
  }
  return '进行中';
});

function playerName(p: string): string {
  return p === 'Red' ? '红方' : '黑方';
}

function describeBoardChange(prev: GameState, next: GameState): string {
  const lines: string[] = [];
  prev.board.forEach((slot, idx) => {
    const now = next.board[idx];
    if (slot === now) return;
    const nowText = pieceText(now) || (now === 'Hidden' ? '?' : '');
    if (slot === 'Hidden') {
      lines.push(`位置${idx} 翻开 → ${nowText}`);
    } else if (now === 'Empty') {
      lines.push(`位置${idx} ${pieceText(slot)} 移出`);
    } else if (slot === 'Empty') {
      lines.push(`位置${idx} 落子 ${nowText}`);
    } else {
      lines.push(`位置${idx} ${nowText} 吃掉 ${pieceText(slot)}`);
    }
  });
  return lines.join('；') || '局面变化';
}

function logMove(prev: GameState, next: GameState) {
  const actor = playerName(next.current_player === 'Red' ? 'Black' : 'Red');
  appendLog(`第${next.move_counter}步 [${actor}] ${describeBoardChange(prev, next)}`);
}

async function cachedMoveAction(fromSq: number, toSq: number): Promise<number | null> {
  const key = `${fromSq}-${toSq}`;
  if (actionIndexCache.has(key)) return actionIndexCache.get(key)!;
  try {
    const action = await api.getMoveAction(fromSq, toSq);
    actionIndexCache.set(key, action);
    return action;
  } catch (err) {
    console.error('get_move_action failed', err);
    actionIndexCache.set(key, null);
    return null;
  }
}

function legalReveal(idx: number): boolean {
  return store.state?.action_masks[idx] === 1;
}

function isSelectable(idx: number | null): boolean {
  const s = store.state;
  if (!s || idx == null) return false;
  const slot = s.board[idx];
  return isRevealed(slot) && slotPlayer(slot) === s.current_player;
}

async function computeMoveHighlights(from: number): Promise<Map<number, MoveHighlight>> {
  const s = store.state;
  const result = new Map<number, MoveHighlight>();
  if (!s || from == null) return result;
  const tasks = s.board.map(async (target, idx) => {
    if (idx === from) return null;
    const action = await cachedMoveAction(from, idx);
    if (action == null || s.action_masks[action] !== 1) return null;
    if (target === 'Empty') return { idx, type: 'move' as const };
    if (target === 'Hidden') return null;
    if (slotPlayer(target) !== s.current_player) return { idx, type: 'capture' as const };
    return null;
  });
  for (const entry of await Promise.all(tasks)) {
    if (entry) result.set(entry.idx, entry);
  }
  return result;
}

async function refreshHighlights() {
  store.moveHighlights =
    store.selectedSquare !== null
      ? await computeMoveHighlights(store.selectedSquare)
      : new Map();
}

function checkGameOver(result: { terminated: boolean; truncated: boolean; winner: number }) {
  if (!result.terminated && !result.truncated) return;
  const msg =
    '游戏结束！' +
    (result.winner === 1 ? ' 红方获胜！' : result.winner === -1 ? ' 黑方获胜！' : ' 平局！');
  appendLog(msg);
  toast.success(msg, 5000);
}

async function applyStep(prev: GameState, result: { state: GameState } & { terminated: boolean; truncated: boolean; winner: number }) {
  store.lastResult = result;
  logMove(prev, result.state);
  store.state = result.state;
  store.selectedSquare = null;
  store.moveHighlights = new Map();
  checkGameOver(result);
}

async function maybeBotTurn() {
  try {
    const opp = await api.getOpponentType();
    if (opp === 'PvP') return;
    store.busy = true;
    const prev = store.state;
    if (!prev) return;
    const result = await api.botMove();
    await applyStep(prev, result);
  } catch (e) {
    console.error('bot_move skipped or failed:', e);
    toast.error('电脑行动失败: ' + e);
  } finally {
    store.busy = false;
  }
}

async function doMove(action: number) {
  if (!store.state || store.busy) return;
  const prev = store.state;
  store.busy = true;
  try {
    const result = await api.stepGame(action);
    await applyStep(prev, result);
    if (!(result.terminated || result.truncated)) {
      await maybeBotTurn();
    }
  } catch (e) {
    toast.error('操作失败: ' + e);
  } finally {
    store.busy = false;
  }
}

async function onSquareClick(idx: number) {
  const s = store.state;
  if (!s || store.busy) return;
  const slot = s.board[idx];

  if (store.selectedSquare === null) {
    if (slot === 'Hidden') {
      if (legalReveal(idx)) {
        await doMove(idx);
      } else {
        toast.info('该棋子当前不能翻开', 1500);
      }
    } else if (isRevealed(slot) && slotPlayer(slot) === s.current_player) {
      store.selectedSquare = idx;
      await refreshHighlights();
    }
    return;
  }

  if (idx === store.selectedSquare) {
    store.selectedSquare = null;
    store.moveHighlights = new Map();
  } else if (isRevealed(slot) && slotPlayer(slot) === s.current_player) {
    store.selectedSquare = idx;
    await refreshHighlights();
  } else {
    const action = await cachedMoveAction(store.selectedSquare, idx);
    if (action !== null && s.action_masks[action] === 1) {
      await doMove(action);
    } else {
      toast.info('该棋子无法移动到此处', 1500);
    }
  }
}

function deselect() {
  if (store.selectedSquare !== null) {
    store.selectedSquare = null;
    store.moveHighlights = new Map();
  }
}

async function resetGame(opponent: Opponent, variant: Variant) {
  if (store.busy) return;
  store.selectedSquare = null;
  store.moveHighlights = new Map();
  try {
    appendLog(`开始新游戏（变体 ${variant}，对手 ${opponent}）`);
    store.state = await api.resetGame(opponent, variant);
    store.lastResult = null;
  } catch (e) {
    console.error('Reset game failed:', e);
    toast.error('重置游戏失败: ' + e);
  }
}

async function loadInitialState() {
  try {
    store.state = await api.getGameState();
  } catch (e) {
    console.error('Failed to load initial state:', e);
    toast.error('加载初始状态失败: ' + e, 5000);
  }
}

export function useGame() {
  return {
    store,
    statusText,
    loadInitialState,
    resetGame,
    onSquareClick,
    deselect,
    isSelectable,
  };
}
