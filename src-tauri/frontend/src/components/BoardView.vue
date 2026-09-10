<script setup lang="ts">
import { computed } from 'vue';
import { useGame } from '../composables/useGame';
import { pieceText, slotPlayer, variantDims } from '../domain/pieces';

const { store, onSquareClick } = useGame();

const dims = computed(() =>
  variantDims(store.state?.variant ?? 'dark', store.state?.board.length ?? 32),
);

const gridStyle = computed(() => ({
  gridTemplateColumns: `repeat(${dims.value.cols}, 1fr)`,
  gridTemplateRows: `repeat(${dims.value.rows}, 1fr)`,
  aspectRatio: `${dims.value.cols} / ${dims.value.rows}`,
}));

function cellClass(idx: number): string[] {
  const s = store.state;
  const slot = s?.board[idx] ?? 'Empty';
  const classes = ['chess-cell'];
  if (slot === 'Hidden') classes.push('hidden');
  else if (slot === 'Empty') classes.push('empty');
  else classes.push(slotPlayer(slot) === 'Red' ? 'red' : 'black');

  if (store.selectedSquare === idx) classes.push('selected');
  if (slot === 'Hidden' && s?.action_masks[idx] === 1) classes.push('legal-reveal');
  const hl = store.moveHighlights.get(idx);
  if (hl) classes.push(hl.type === 'capture' ? 'legal-capture' : 'legal-move');
  return classes;
}
</script>

<template>
  <div class="board-wrap" :class="{ busy: store.busy }">
    <div
      v-if="store.state"
      class="chess-board"
      :class="{ 'board-square': dims.cols === dims.rows }"
      :style="gridStyle"
    >
      <div
        v-for="(slot, idx) in store.state.board"
        :key="idx"
        :class="cellClass(idx)"
        @click="onSquareClick(idx)"
      >
        {{ pieceText(slot) }}
      </div>
    </div>
    <div v-else class="board-empty">加载中…</div>
    <div v-if="store.busy" class="busy-banner">电脑思考中…</div>
  </div>
</template>
