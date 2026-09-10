import { reactive, ref } from 'vue';
import { api } from '../api/client';
import type { ModelEntry, Opponent, Variant } from '../api/types';
import { appendLog } from './useLogs';
import { useToast } from './useToast';

const toast = useToast();

const settings = reactive({
  variant: 'dark' as Variant,
  opponent: 'PvP' as Opponent,
  engineLevel: 300000,
  mctsIters: 200,
  nnueDepth: 8,
  nnueBudget: 200000,
});

const ptModels = ref<ModelEntry[]>([]);
const nnueModels = ref<ModelEntry[]>([]);
const modelsLoading = ref(false);

async function refreshModels() {
  modelsLoading.value = true;
  try {
    const models = await api.listModels();
    ptModels.value = models.filter((m) => !m.path.toLowerCase().endsWith('.nnue'));
    nnueModels.value = models.filter((m) => m.path.toLowerCase().endsWith('.nnue'));
  } catch (e) {
    console.error('list_models failed:', e);
    toast.error('加载模型列表失败: ' + e);
  } finally {
    modelsLoading.value = false;
  }
}

async function applyEngineBudget(): Promise<boolean> {
  try {
    const result = await api.setEngineBudget(settings.engineLevel);
    appendLog(`强引擎节点预算已设置为 ${result}`);
    toast.success(`强引擎节点预算已设置为 ${result}`);
    return true;
  } catch (e) {
    toast.error('设置失败: ' + e);
    return false;
  }
}

async function applyMctsIters(): Promise<boolean> {
  try {
    const result = await api.setMctsIterations(settings.mctsIters);
    settings.mctsIters = result;
    appendLog(`MCTS 搜索次数已设置为 ${result}`);
    toast.success('MCTS 搜索次数已设置为 ' + result);
    return true;
  } catch (e) {
    toast.error('设置失败: ' + e);
    return false;
  }
}

async function applyNnue(): Promise<boolean> {
  try {
    const [d, b] = await Promise.all([
      api.setNnueDepth(settings.nnueDepth),
      api.setNnueBudget(settings.nnueBudget),
    ]);
    settings.nnueDepth = d;
    settings.nnueBudget = b;
    appendLog(`NNUE 设置已应用：深度 ${d} / 节点预算 ${b}`);
    toast.success(`NNUE 设置已应用：深度 ${d} / 节点预算 ${b}`);
    return true;
  } catch (e) {
    toast.error('设置失败: ' + e);
    return false;
  }
}

async function loadModel(path: string): Promise<boolean> {
  try {
    const result = await api.loadModel(path);
    appendLog(result);
    toast.success('模型加载成功：' + result);
    return true;
  } catch (e) {
    toast.error('模型加载失败：' + e, 5000);
    return false;
  }
}

export function useSettings() {
  return {
    settings,
    ptModels,
    nnueModels,
    modelsLoading,
    refreshModels,
    applyEngineBudget,
    applyMctsIters,
    applyNnue,
    loadModel,
  };
}
