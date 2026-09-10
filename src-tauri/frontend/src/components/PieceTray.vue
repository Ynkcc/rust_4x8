<script setup lang="ts">
import { computed } from 'vue';
import type { Player } from '../api/types';
import { PIECE_META, PIECE_TYPE_ORDER, countByType } from '../domain/pieces';

const props = defineProps<{
  player: Player;
  pieces: string[] | undefined;
  dimWhenZero?: boolean;
}>();

const counts = computed(() => countByType(props.pieces));
</script>

<template>
  <div class="tray">
    <div
      v-for="type in PIECE_TYPE_ORDER"
      :key="type"
      class="tray-item"
      :class="{
        'no-loss': dimWhenZero && counts[type] === 0,
        'has-loss': dimWhenZero && counts[type] > 0,
      }"
    >
      <span
        class="tray-icon"
        :class="player === 'Red' ? 'red' : 'black'"
        :data-count="counts[type] > 1 ? counts[type] : undefined"
      >
        {{ PIECE_META[type][player === 'Red' ? 'red' : 'black'] }}
      </span>
    </div>
  </div>
</template>
