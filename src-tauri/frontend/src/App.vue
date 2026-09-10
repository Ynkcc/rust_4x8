<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { useGame } from './composables/useGame';
import { useSettings } from './composables/useSettings';
import BoardView from './components/BoardView.vue';
import BitboardPanel from './components/BitboardPanel.vue';
import ControlPanel from './components/ControlPanel.vue';
import LogPanel from './components/LogPanel.vue';
import MctsTreePanel from './components/MctsTreePanel.vue';
import PieceTray from './components/PieceTray.vue';
import StatusPanel from './components/StatusPanel.vue';
import ToastHost from './components/ToastHost.vue';

const { store, loadInitialState, deselect } = useGame();
const { refreshModels } = useSettings();

const activeDrawer = ref<'none' | 'bitboard' | 'mcts'>('none');
const overlayVisible = computed(() => activeDrawer.value !== 'none');

function closeDrawer() {
  activeDrawer.value = 'none';
}

function onKeydown(evt: KeyboardEvent) {
  if (evt.key !== 'Escape') return;
  if (activeDrawer.value !== 'none') {
    closeDrawer();
    return;
  }
  deselect();
}

onMounted(async () => {
  window.addEventListener('keydown', onKeydown);
  await loadInitialState();
  await refreshModels();
});
onUnmounted(() => window.removeEventListener('keydown', onKeydown));
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <h1>暗棋 4x8</h1>
      <div class="header-actions">
        <button class="pill" @click="activeDrawer = activeDrawer === 'bitboard' ? 'none' : 'bitboard'">
          通道状态
        </button>
        <button class="pill" @click="activeDrawer = activeDrawer === 'mcts' ? 'none' : 'mcts'">
          搜索树
        </button>
      </div>
    </header>

    <main class="app-main">
      <aside class="left-column">
        <ControlPanel />
      </aside>

      <section class="center-column">
        <div class="board-zone">
          <div class="tray-strip">
            <span class="tray-strip-title">黑方阵亡</span>
            <PieceTray player="Black" :pieces="store.state?.dead_black" :dim-when-zero="true" />
          </div>

          <BoardView />

          <div class="tray-strip">
            <span class="tray-strip-title">红方阵亡</span>
            <PieceTray player="Red" :pieces="store.state?.dead_red" :dim-when-zero="true" />
          </div>
        </div>

        <div class="insights">
          <div class="panel-card">
            <h3>隐藏棋子</h3>
            <div class="tray-row">
              <span class="tray-strip-title">红</span>
              <PieceTray player="Red" :pieces="store.state?.hidden_red" :dim-when-zero="true" />
            </div>
            <div class="tray-row">
              <span class="tray-strip-title">黑</span>
              <PieceTray player="Black" :pieces="store.state?.hidden_black" :dim-when-zero="true" />
            </div>
          </div>
          <StatusPanel />
        </div>
      </section>

      <aside class="right-column">
        <LogPanel />
      </aside>
    </main>

    <Teleport to="body">
      <div class="overlay" :class="{ active: overlayVisible }" @click="closeDrawer"></div>
      <Transition name="drawer">
        <BitboardPanel v-if="activeDrawer === 'bitboard'" @close="closeDrawer" />
      </Transition>
      <Transition name="drawer">
        <MctsTreePanel v-if="activeDrawer === 'mcts'" @close="closeDrawer" />
      </Transition>
    </Teleport>

    <ToastHost />
  </div>
</template>
