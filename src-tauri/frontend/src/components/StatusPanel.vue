<script setup lang="ts">
import { computed } from 'vue';
import { useGame } from '../composables/useGame';
import { maxHp } from '../domain/pieces';

const { store, statusText } = useGame();

const variantMaxHp = computed(() => maxHp(store.state?.variant ?? 'dark'));

function hpPct(hp: number): string {
  return Math.max(0, Math.min(100, Math.round((hp / variantMaxHp.value) * 100))) + '%';
}
</script>

<template>
  <div class="status-panel">
    <div class="player-row">
      <span
        class="player-indicator"
        :class="{ 'black-turn': store.state?.current_player === 'Black' }"
      >
        <span class="indicator-dot"></span>
        <span>{{ store.state?.current_player === 'Black' ? '黑方' : '红方' }}</span>
      </span>
      <span class="status-text">{{ statusText }}</span>
    </div>
    <div class="hp-row">
      <span class="hp-label red-label">红方</span>
      <div class="hp-bar-outer">
        <div class="hp-bar-fill hp-red-fill" :style="{ width: hpPct(store.state?.hp_red ?? 0) }"></div>
      </div>
      <span class="hp-value">{{ store.state?.hp_red ?? '—' }}</span>
    </div>
    <div class="hp-row">
      <span class="hp-label black-label">黑方</span>
      <div class="hp-bar-outer">
        <div class="hp-bar-fill hp-black-fill" :style="{ width: hpPct(store.state?.hp_black ?? 0) }"></div>
      </div>
      <span class="hp-value">{{ store.state?.hp_black ?? '—' }}</span>
    </div>
    <div class="counter-row">
      <span>回合：{{ store.state?.move_counter ?? '—' }}</span>
      <span>总步数：{{ store.state?.total_step_counter ?? '—' }}</span>
      <span>未吃子：{{ store.state?.move_counter ?? '—' }}</span>
    </div>
  </div>
</template>
