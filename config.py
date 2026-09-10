from pathlib import Path
from typing import Final

# Base 路径，指向项目根目录
BASE_DIR = Path(__file__).resolve().parent
RED_SCOUT_MAX_COUNT = 50
RED_SCOUT_DEFAULT_COUNT: Final[int] = 20

# ADB 连接的默认设备 IP 地址
ADB_SERIAL = "127.0.0.1:5555"

# Bundled adb executable inside the repository.
ADB_EXE = BASE_DIR / "tools" / "platform-tools" / "adb.exe"

# 默认控制的游戏包名
GAME_PACKAGE_NAME = "com.tencent.tmgp.supercell.boombeach"

# 模板图片目录和截图保存目录
TEMPLATE_DIR = BASE_DIR / "template"
SCREENSHOT_DIR = BASE_DIR / "_debug" / "screenshots"
LOG_DIR = BASE_DIR / "_debug" / "logs"
LOG_FILE = LOG_DIR / "bbma.log"
OUTPUT_DIR = BASE_DIR / "outputs"
MAX_PROBE_SAMPLE_DIRS: Final[int] = 20
MAX_RED_SCOUT_SAMPLE_DIRS: Final[int] = 10
MAX_SCREENSHOT_STORAGE_BYTES: Final[int] = 500 * 1024 * 1024


# 目前支持的最大关卡
MAX_LEVEL: Final[int] = 70

# 二值法识图模式（开局潜艇格识别）：
#   "only"  = 只用二值法（board_cell_model.json）识别，忽略残骸/模板/特征启动视觉
#   "merge" = 二值法结果叠加进现有启动命中
#   ""      = 关闭二值法识图（当前：全部二值法已停用）
# 可用环境变量 BBMA_USE_BOARD_RECOGNIZER 覆盖。
BOARD_RECOGNIZER_MODE: Final[str] = ""

# 严格二值法模式：True 时开局识图**只用二值法结果**，丢弃其它所有开局视觉
# （模板/残骸/特征/侧边栏/红标记）。默认 False = 保留其它开局视觉。
# 可用环境变量 BBMA_BOARD_STRICT_ONLY 覆盖（"1"/"true" 开）。
BOARD_RECOGNIZER_STRICT_ONLY: Final[bool] = False

# 开局是否用 sprite 级潜艇/残骸检测器（utils/submarine_detector.py，棋盘四边形 ROI）：
#   True  = 运行，并把命中的潜艇格/碎片格叠加进开局命中
# 可用环境变量 BBMA_USE_SUBMARINE_DETECTOR 覆盖（"1"/"true" 强制开，"0"/"false" 强制关）。
USE_SUBMARINE_DETECTOR: Final[bool] = False

# 命中/揭示判定方法：
#   "binarize" = 二值法前后帧差分（打点后新出现"潜艇内容"=命中）
#   "classic"  = 原色彩/模板/侧边栏判定（当前：二值法已停用）
# 可用环境变量 BBMA_HIT_METHOD 覆盖。
HIT_METHOD: Final[str] = "classic"

# 二值法在"攻击后判定"里的角色（HIT_METHOD="binarize" 时才生效）：
#   True  = 以二值法前后帧差分结果**为准**（不再只做补充）
#   False = 二值法只把未命中补成命中（原行为，绝不把命中改成未命中）
# 可用环境变量 BBMA_HIT_BINARIZE_PRIMARY 覆盖。
HIT_BINARIZE_PRIMARY: Final[bool] = False

# 自动识别不可用时使用的默认回退关卡
DEFAULT_LEVEL: Final[int] = 2

# Automatic level recognition from save_points/imgs reference screenshots.
AUTO_DETECT_LEVEL: Final[bool] = True
REQUIRE_CONFIDENT_LEVEL_DETECTION: Final[bool] = True
LEVEL_REFERENCE_DIR = BASE_DIR / "save_points" / "imgs"
LEVEL_DETECTION_MIN_SCORE: Final[float] = 0.62
LEVEL_DETECTION_MIN_MARGIN: Final[float] = 0.08

# 第 11 海域及以上关卡使用的默认潜艇长度列表
DEFAULT_SUBMARINES: Final[tuple[int, ...]] = (2, 2, 3, 4, 5)

# 固定关卡对应的潜艇长度列表，供前 10 个关卡使用
SPECIAL_SUBMARINES: Final[dict[int, tuple[int, ...]]] = {
    1:  (3,),
    2:  (2, 2),
    3:  (2, 2, 3),
    4:  (2, 3, 4),
    5:  (2, 3, 3, 4),
    6:  (2, 2, 3, 3, 5),
    7:  (2, 2, 3, 3, 4, 5),
    8:  (2, 2, 3, 3, 4, 4, 5),
    9:  (2, 3, 3, 4, 4, 5),
    10: (2, 2, 3, 4, 4, 5),
}

# 固定关卡对应的菱形网格边长
LEVEL_GRID_SIZES: Final[dict[int, int]] = {
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 8,
    7: 9,
    8: 10,
    9: 10,
    10: 10,
    **{
        level: 10
        for level in range(11, MAX_LEVEL + 1)
    },

}

# Level 对应的潜艇长度列表
SUBMARINES: Final[dict[int, tuple[int, ...]]] = {
    **SPECIAL_SUBMARINES,
    **{
        level: DEFAULT_SUBMARINES
        for level in range(11, MAX_LEVEL + 1)
    },
}


# 是否优先使用人工校准后的固定点位
USE_SAVED_POINTS = True
SAVED_POINTS_FILE = BASE_DIR / "save_points" / "points.json"

# 默认的截图文件名和模板匹配的默认阈值
DEFAULT_SCREENSHOT_NAME = "screen.png"
DEFAULT_MATCH_THRESHOLD = 0.85
DEFAULT_TEMPLATE_SHAPE_WEIGHT = 0.9
DEFAULT_TEMPLATE_SHAPE_POWER = 3.0

# 日志级别，可选 DEBUG、INFO、WARNING、ERROR
LOG_LEVEL = "INFO"
