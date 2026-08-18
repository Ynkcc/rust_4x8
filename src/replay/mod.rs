// src/replay.rs - 从对局记录还原人类可读的棋谱文字描述
//
// 设计说明：
// - 阵营由手数奇偶决定：第 i 手（0 基）i%2==0 → 红方、i%2==1 → 黑方，
//   因此无需外部手动传入己方颜色；
// - 用 boards/scalars 逐手还原棋盘 → 重建 DarkChessEnv → 重新生成 action_masks，
//   与记录中的 action_masks 逐元素断言一致；
// - 断言记录的 actions[i] 一定在合法掩码内；
// - 输出中文棋谱描述，动作带坐标，如：红马(0,a)->黑兵(1,b)。
//
// 变体支持：所有解码函数均由 `GameConfig` 驱动，支持 4x8 / 4x2 / 4x4。
// `describe_record` 保持 4x8 默认（向后兼容），`describe_record_with_config`
// 接受任意变体配置。

mod decode;
mod describe;
mod scalar;
mod util;

pub use decode::{decode_board, decode_board_with_config};
pub use describe::{describe_record, describe_record_with_config};
pub use scalar::{
    ScalarDecodeResult, decode_scalar_state, format_scalar_state, survival_to_dead_vec,
};
