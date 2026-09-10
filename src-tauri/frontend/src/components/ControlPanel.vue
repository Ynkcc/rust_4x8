<script setup lang="ts">
import { ref } from 'vue';
import type { Opponent, Variant } from '../api/types';
import { useGame } from '../composables/useGame';
import { useSettings } from '../composables/useSettings';
import { useToast } from '../composables/useToast';

const { store, resetGame } = useGame();
const { settings, ptModels, nnueModels, refreshModels, applyEngineBudget, applyMctsIters, applyNnue, loadModel } =
  useSettings();
const toast = useToast();

const OPPONENTS: { value: Opponent; label: string }[] = [
  { value: 'PvP', label: '本地双人 (PvP)' },
  { value: 'Random', label: '电脑 (随机)' },
  { value: 'RevealFirst', label: '电脑 (优先翻棋)' },
  { value: 'Engine', label: '电脑 (强引擎)' },
  { value: 'MctsDL', label: '电脑 (MCTS+DL)' },
  { value: 'MctsOnnx', label: '电脑 (MCTS+ONNX)' },
  { value: 'Nnue', label: '电脑 (NNUE)' },
];

const VARIANTS: { value: Variant; label: string }[] = [
  { value: 'dark', label: '暗棋 (4x8)' },
  { value: '4x4', label: '暗棋 (4x4)' },
  { value: 'mini', label: '迷你暗棋 (4x2)' },
];

const ENGINE_LEVELS = [
  { value: 50000, label: '轻量（5万节点）' },
  { value: 300000, label: '标准（30万节点）' },
  { value: 1000000, label: '深度（100万节点）' },
  { value: 3000000, label: '极深（300万节点）' },
];

const ptPathInput = ref('');
const nnuePathInput = ref('');
const ptSelected = ref('');
const nnueSelected = ref('');

function showEngineSettings(): boolean {
  return settings.opponent === 'Engine';
}
function showMctsSettings(): boolean {
  return settings.opponent === 'MctsDL' || settings.opponent === 'MctsOnnx';
}
function showNnueSettings(): boolean {
  return settings.opponent === 'Nnue';
}

function onNewGame() {
  resetGame(settings.opponent, settings.variant);
}

async function onLoadPtModel() {
  const finalPath = ptPathInput.value.trim() || ptSelected.value;
  if (!finalPath) {
    toast.error('请先选择模型（或手动输入模型路径）');
    return;
  }
  if (await loadModel(finalPath)) {
    settings.opponent = finalPath.toLowerCase().endsWith('.onnx') ? 'MctsOnnx' : 'MctsDL';
  }
}

async function onLoadNnueModel() {
  const finalPath = nnuePathInput.value.trim() || nnueSelected.value;
  if (!finalPath) {
    toast.error('请先选择 .nnue 模型（或手动输入模型路径）');
    return;
  }
  if (await loadModel(finalPath)) {
    settings.opponent = 'Nnue';
  }
}
</script>

<template>
  <div class="control-panel">
    <section class="panel-card">
      <h3>对局设置</h3>
      <label class="field">
        <span>变体</span>
        <select v-model="settings.variant">
          <option v-for="v in VARIANTS" :key="v.value" :value="v.value">{{ v.label }}</option>
        </select>
      </label>
      <label class="field">
        <span>对手</span>
        <select v-model="settings.opponent">
          <option v-for="o in OPPONENTS" :key="o.value" :value="o.value">{{ o.label }}</option>
        </select>
      </label>
      <button class="primary" :disabled="store.busy" @click="onNewGame">开始新游戏</button>
    </section>

    <section v-if="showEngineSettings()" class="panel-card">
      <h3>强引擎设置</h3>
      <label class="field">
        <span>难度</span>
        <select v-model.number="settings.engineLevel">
          <option v-for="l in ENGINE_LEVELS" :key="l.value" :value="l.value">{{ l.label }}</option>
        </select>
      </label>
      <button @click="applyEngineBudget">应用难度</button>
      <p class="hint">αβ + Star1 + 置换表 + 迭代加深，节点预算越高越强、耗时越长。</p>
    </section>

    <section v-if="showMctsSettings()" class="panel-card">
      <h3>MCTS+DL 设置（TorchScript / ONNX）</h3>
      <label class="field">
        <span>模型</span>
        <select v-model="ptSelected">
          <option v-if="ptModels.length === 0" value="">未找到 .pt / .onnx 模型</option>
          <option v-for="m in ptModels" :key="m.path" :value="m.path">{{ m.path }}</option>
        </select>
      </label>
      <div class="btn-row">
        <button :disabled="store.busy" @click="refreshModels">刷新列表</button>
        <button :disabled="store.busy" @click="onLoadPtModel">加载模型</button>
      </div>
      <label class="field">
        <span>路径</span>
        <input v-model="ptPathInput" type="text" placeholder="或手动输入 .pt / .onnx 路径" />
      </label>
      <label class="field">
        <span>搜索次数</span>
        <input v-model.number="settings.mctsIters" type="number" min="1" />
      </label>
      <button @click="applyMctsIters">应用搜索次数</button>
      <p class="hint">先加载模型（.pt 走 MCTS+DL，.onnx 走 MCTS+ONNX），再选择对应对手开始新游戏。</p>
    </section>

    <section v-if="showNnueSettings()" class="panel-card">
      <h3>NNUE 设置（Expectimax + .nnue）</h3>
      <label class="field">
        <span>模型</span>
        <select v-model="nnueSelected">
          <option v-if="nnueModels.length === 0" value="">未找到 .nnue 模型</option>
          <option v-for="m in nnueModels" :key="m.path" :value="m.path">{{ m.path }}</option>
        </select>
      </label>
      <div class="btn-row">
        <button :disabled="store.busy" @click="refreshModels">刷新列表</button>
        <button :disabled="store.busy" @click="onLoadNnueModel">加载模型</button>
      </div>
      <label class="field">
        <span>路径</span>
        <input v-model="nnuePathInput" type="text" placeholder="或手动输入 .nnue 路径" />
      </label>
      <label class="field">
        <span>搜索深度</span>
        <input v-model.number="settings.nnueDepth" type="number" min="1" />
      </label>
      <label class="field">
        <span>节点预算</span>
        <input v-model.number="settings.nnueBudget" type="number" min="1" />
      </label>
      <button @click="applyNnue">应用设置</button>
      <p class="hint">模型特征维度需与所选变体匹配。</p>
    </section>
  </div>
</template>
