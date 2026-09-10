/** 后端 Tauri command 返回的数据结构（与 src-tauri/src/main.rs 中 serde 序列化一一对应）。 */

export type Player = 'Red' | 'Black';
export type Slot = 'Empty' | 'Hidden' | string; // 明子格式: "R_Sol" / "B_Gen" ...

export interface GameState {
  board: Slot[];
  current_player: Player;
  move_counter: number;
  total_step_counter: number;
  dead_red: string[];
  dead_black: string[];
  hidden_red: string[];
  hidden_black: string[];
  action_masks: number[];
  reveal_probabilities: number[];
  bitboards: Record<string, boolean[]>;
  hp_red: number;
  hp_black: number;
  variant: 'dark' | '4x4' | 'mini';
}

export interface StepResult {
  state: GameState;
  terminated: boolean;
  truncated: boolean;
  winner: number;
}

export type Opponent =
  | 'PvP'
  | 'Random'
  | 'RevealFirst'
  | 'Engine'
  | 'MctsDL'
  | 'MctsOnnx'
  | 'Nnue';

export type Variant = 'dark' | '4x4' | 'mini';

export interface ModelEntry {
  name: string;
  path: string;
}

export interface MctsEdge {
  child_id: number;
  action: number;
  prior: number;
  logit: number;
  n: number;
  q: number;
  health_q: number;
  is_chance: boolean;
  chance_prob: number;
}

export interface MctsNodeSummary {
  id: number;
  n: number;
  q: number;
  health_q: number;
  player: Player;
  is_chance: boolean;
  is_terminal: boolean;
  is_expanded: boolean;
  edge_count: number;
}

export interface MctsRootInfo {
  root: MctsNodeSummary;
  chosen_action: number;
}

export interface MctsNodeDetail {
  id: number;
  prior: number;
  logit: number;
  n: number;
  q: number;
  health_q: number;
  initial_value: number;
  player: Player;
  is_chance: boolean;
  is_terminal: boolean;
  is_expanded: boolean;
  child_count: number;
  outcome_count: number;
}
