import { invoke } from '@tauri-apps/api/core';
import type {
  GameState,
  MctsEdge,
  MctsNodeDetail,
  MctsRootInfo,
  ModelEntry,
  Opponent,
  StepResult,
  Variant,
} from './types';

export const api = {
  resetGame(opponent: Opponent, variant: Variant) {
    return invoke<GameState>('reset_game', { opponent, variant });
  },
  stepGame(action: number) {
    return invoke<StepResult>('step_game', { action });
  },
  botMove() {
    return invoke<StepResult>('bot_move');
  },
  getGameState() {
    return invoke<GameState>('get_game_state');
  },
  getOpponentType() {
    return invoke<Opponent>('get_opponent_type');
  },
  getMoveAction(fromSq: number, toSq: number) {
    return invoke<number | null>('get_move_action', { fromSq, toSq });
  },
  listModels() {
    return invoke<ModelEntry[]>('list_models');
  },
  loadModel(path: string) {
    return invoke<string>('load_model', { path });
  },
  setMctsIterations(iters: number) {
    return invoke<number>('set_mcts_iterations', { iters });
  },
  setEngineBudget(budget: number) {
    return invoke<number>('set_engine_budget', { budget });
  },
  setNnueDepth(depth: number) {
    return invoke<number>('set_nnue_depth', { depth });
  },
  setNnueBudget(budget: number) {
    return invoke<number>('set_nnue_budget', { budget });
  },
  mctsGetRoot() {
    return invoke<MctsRootInfo>('mcts_get_root');
  },
  mctsGetChildren(nodeId: number) {
    return invoke<MctsEdge[]>('mcts_get_children', { nodeId });
  },
  mctsGetNodeDetail(nodeId: number) {
    return invoke<MctsNodeDetail>('mcts_get_node_detail', { nodeId });
  },
  mctsSearch() {
    return invoke<MctsRootInfo>('mcts_search');
  },
};
