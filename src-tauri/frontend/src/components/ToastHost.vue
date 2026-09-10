<script setup lang="ts">
import { useToast } from '../composables/useToast';
const { toasts } = useToast();
</script>

<template>
  <Teleport to="body">
    <div class="toast-container">
      <div v-for="t in toasts" :key="t.id" class="toast" :class="`toast-${t.type}`">
        {{ t.text }}
      </div>
    </div>
  </Teleport>
</template>
