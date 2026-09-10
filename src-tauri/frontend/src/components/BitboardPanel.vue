<script setup lang="ts">
import { computed } from 'vue';
import { useGame } from '../composables/useGame';
import { BITBOARD_ORDER } from '../domain/pieces';

const { store } = useGame();

defineEmits<{ close: [] }>();

const entries = computed(() => {
  const bbs = store.state?.bitboards;
  if (!bbs) return [];
  return BITBOARD_ORDER.filter(({ key }) => Array.isArray(bbs[key])).map(({ key, label }) => ({
    label,
    values: bbs[key],
  }));
});

function gridStyle(len: number) {
  const cols = len === 8 ? 2 : len === 16 ? 4 : len === 32 ? 8 : len / 4;
  return { gridTemplateColumns: `repeat(${cols}, 1fr)`, gridTemplateRows: 'repeat(4, 1fr)' };
}
</script>

<template>
  <aside class="sidebar">
    <header class="sidebar-header">
      <h3>通道状态 (Bitboards)</h3>
      <button class="icon-button" aria-label="关闭" @click="$emit('close')">×</button>
    </header>
    <div class="sidebar-body bitboard-list">
      <div v-if="entries.length === 0" class="bitboard-empty">暂无通道数据</div>
      <div v-for="entry in entries" :key="entry.label" class="bb-wrapper">
        <div class="bb-label">{{ entry.label }}</div>
        <div class="bb-grid" :style="gridStyle(entry.values.length)">
          <div
            v-for="(active, i) in entry.values"
            :key="i"
            class="bb-cell"
            :class="{ active }"
          ></div>
        </div>
      </div>
    </div>
  </aside>
</template>
