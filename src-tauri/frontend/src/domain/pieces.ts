import type { Slot } from '../api/types';

export const PIECE_TYPE_ORDER = [
  'General',
  'Advisor',
  'Elephant',
  'Chariot',
  'Horse',
  'Cannon',
  'Soldier',
] as const;

/** 后端短码（R_Sol / B_Gen …）→ 全名（dead_pieces / PIECE_TYPE_ORDER 使用） */
const SHORT_TO_TYPE: Record<string, string> = {
  Sol: 'Soldier',
  Can: 'Cannon',
  Hor: 'Horse',
  Car: 'Chariot',
  Ele: 'Elephant',
  Adv: 'Advisor',
  Gen: 'General',
};

/** 全名 → 红黑文字 */
export const PIECE_META: Record<string, { red: string; black: string }> = {
  General: { red: '帥', black: '將' },
  Advisor: { red: '仕', black: '士' },
  Elephant: { red: '相', black: '象' },
  Chariot: { red: '車', black: '車' },
  Horse: { red: '馬', black: '馬' },
  Cannon: { red: '炮', black: '砲' },
  Soldier: { red: '兵', black: '卒' },
};

export const BITBOARD_ORDER: { key: string; label: string }[] = [
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

export function isRevealed(slot: Slot): boolean {
  return slot !== 'Empty' && slot !== 'Hidden';
}

export function slotPlayer(slot: Slot): 'Red' | 'Black' | null {
  if (slot.startsWith('R_')) return 'Red';
  if (slot.startsWith('B_')) return 'Black';
  return null;
}

export function pieceText(slot: Slot): string {
  if (slot === 'Empty') return '';
  if (slot === 'Hidden') return '?';
  const player = slotPlayer(slot);
  const type = SHORT_TO_TYPE[slot.substring(2)];
  const meta = type ? PIECE_META[type] : undefined;
  if (!meta) return '';
  return player === 'Red' ? meta.red : meta.black;
}

export function countByType(names: string[] | undefined): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const t of PIECE_TYPE_ORDER) counts[t] = 0;
  for (const name of names ?? []) {
    if (name in counts) counts[name] += 1;
  }
  return counts;
}

export function variantDims(variant: string, boardLen: number): { rows: number; cols: number } {
  if (variant === 'mini' || boardLen === 8) return { rows: 4, cols: 2 };
  if (variant === '4x4' || boardLen === 16) return { rows: 4, cols: 4 };
  return { rows: 4, cols: 8 };
}

export function maxHp(variant: string): number {
  return variant === 'mini' ? 47 : 60;
}
