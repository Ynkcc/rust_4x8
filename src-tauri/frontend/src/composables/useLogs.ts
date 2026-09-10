import { reactive } from 'vue';

export interface LogItem {
  id: number;
  time: string;
  text: string;
}

let nextId = 0;
const logs = reactive<LogItem[]>([]);

export function appendLog(text: string) {
  const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  logs.push({ id: ++nextId, time, text });
  if (logs.length > 400) logs.shift();
}

export function clearLogs() {
  logs.length = 0;
}

export function useLogs() {
  return logs;
}
