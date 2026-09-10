<script setup lang="ts">
import { nextTick, ref, watch } from 'vue';
import { clearLogs, useLogs } from '../composables/useLogs';

const logs = useLogs();
const listEl = ref<HTMLElement | null>(null);

watch(
  () => logs.length,
  async () => {
    await nextTick();
    listEl.value?.scrollTo({ top: listEl.value.scrollHeight });
  },
);
</script>

<template>
  <div class="log-panel">
    <div class="log-header">
      <h3>对局日志</h3>
      <button class="mini" @click="clearLogs">清空</button>
    </div>
    <div ref="listEl" class="log-list">
      <div v-if="logs.length === 0" class="log-empty">对局记录将显示在此处…</div>
      <div v-for="entry in logs" :key="entry.id" class="log-entry">
        <span class="log-time">{{ entry.time }}</span>
        <span>{{ entry.text }}</span>
      </div>
    </div>
  </div>
</template>
