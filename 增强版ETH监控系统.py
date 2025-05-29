#!/usr/bin/env python3
"""
🚀 增强版ETH监控系统 v8.0.0
🎯 功能: 训练数据展示 + 半月历史数据 + 自动刷新 + 定时重训练
📊 特色: 高精度预测 + 智能交易信号 + 实时监控
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import time
import json
import urllib3
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 导入自动刷新组件
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False
    st.warning("⚠️ streamlit-autorefresh 组件未安装，自动刷新功能不可用")

# 🕒 时区配置 - 强制使用北京时间
BEIJING_TZ = pytz.timezone('Asia/Shanghai')

def get_beijing_time():
    """获取北京时间"""
    return datetime.now(BEIJING_TZ)

def format_beijing_time(dt_obj=None, format_str='%H:%M:%S'):
    """格式化北京时间显示"""
    if dt_obj is None:
        dt_obj = get_beijing_time()
    elif dt_obj.tzinfo is None:
        # 如果传入的是naive datetime，假设它是UTC时间
        dt_obj = pytz.UTC.localize(dt_obj).astimezone(BEIJING_TZ)
    elif dt_obj.tzinfo != BEIJING_TZ:
        # 如果是其他时区，转换为北京时间
        dt_obj = dt_obj.astimezone(BEIJING_TZ)
    
    return dt_obj.strftime(format_str)

# ⚡ 性能优化配置
@st.cache_data(ttl=300, show_spinner=False)  # 缓存5分钟
def cached_api_call(url, params=None):
    """缓存API调用结果"""
    try:
        response = requests.get(url, params=params, timeout=10, verify=False)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)  # 缓存30分钟
def cached_model_training_data(data_hash):
    """缓存模型训练数据处理结果"""
    return None  # 实际的缓存逻辑在具体使用时实现

# 新增机器学习算法导入
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# 页面配置
st.set_page_config(
    page_title="🚀 增强版ETH监控",
    page_icon="💎",
    layout="wide"
)

# 样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .price-display {
        font-size: 3.5rem;
        font-weight: bold;
        color: #4ECDC4;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .success-status {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .warning-status {
        background: linear-gradient(135deg, #ff9a56, #ff6b6b);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    .training-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedETHAPI:
    """增强版ETH API"""
    
    def __init__(self):
        self.current_price = None
        self.api_status = {}
        self.working_apis = []
        
    def _create_optimized_session(self):
        """创建优化的HTTP会话"""
        session = requests.Session()
        session.verify = False
        session.trust_env = False
        session.proxies = {'http': None, 'https': None}
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        return session
    
    def get_current_price(self):
        """获取当前ETH价格"""
        session = self._create_optimized_session()
        
        apis = [
            {
                'name': 'CryptoCompare',
                'url': 'https://min-api.cryptocompare.com/data/pricemultifull?fsyms=ETH&tsyms=USD',
                'parse': lambda data: {
                    'price': float(data['RAW']['ETH']['USD']['PRICE']),
                    'change_24h': float(data['RAW']['ETH']['USD']['CHANGEPCT24HOUR'])
                },
                'timeout': 10
            },
            {
                'name': 'Binance',
                'url': 'https://api.binance.com/api/v3/ticker/24hr?symbol=ETHUSDT',
                'parse': lambda data: {
                    'price': float(data['lastPrice']),
                    'change_24h': float(data['priceChangePercent'])
                },
                'timeout': 15
            }
        ]
        
        successful_data = []
        
        for api in apis:
            try:
                response = session.get(api['url'], timeout=api['timeout'])
                
                if response.status_code == 200:
                    data = response.json()
                    parsed_data = api['parse'](data)
                    price = parsed_data['price']
                    
                    if 1000 <= price <= 10000:
                        successful_data.append({
                            'api': api['name'],
                            'price': price,
                            'change_24h': parsed_data.get('change_24h', 0)
                        })
                        self.api_status[api['name']] = 'success'
                        
                        if len(successful_data) >= 1:
                            break
                    else:
                        self.api_status[api['name']] = 'invalid_price'
                else:
                    self.api_status[api['name']] = f'http_error_{response.status_code}'
                    
            except Exception as e:
                self.api_status[api['name']] = f'error: {str(e)[:30]}'
        
        if not successful_data:
            raise Exception("❌ 无法获取真实ETH价格数据")
        
        final_price = np.median([item['price'] for item in successful_data])
        changes = [item['change_24h'] for item in successful_data if item['change_24h'] != 0]
        avg_change = np.mean(changes) if changes else 0
        
        self.current_price = final_price
        
        return {
            'price': final_price,
            'change_24h': avg_change,
            'api_count': len(successful_data),
            'apis_used': [item['api'] for item in successful_data]
        }
    
    def get_historical_data(self, days=30):
        """获取历史数据 - 增强版支持一个月数据"""
        print(f"📊 正在获取过去{days}天的真实ETH历史数据...")
        
        session = self._create_optimized_session()
        
        try:
            # 使用CryptoCompare获取小时级数据
            response = session.get(
                'https://min-api.cryptocompare.com/data/v2/histohour',
                params={
                    'fsym': 'ETH',
                    'tsym': 'USD',
                    'limit': days * 24,  # 一个月 * 24小时
                    'aggregate': 1
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Data' in data and 'Data' in data['Data']:
                    history_data = data['Data']['Data']
                    
                    historical_data = []
                    for item in history_data:
                        dt = datetime.fromtimestamp(item['time'])
                        price = float(item['close'])
                        volume = float(item['volumeto'])
                        high = float(item['high'])
                        low = float(item['low'])
                        open_price = float(item['open'])
                        
                        historical_data.append({
                            'timestamp': dt,
                            'price': price,
                            'volume': volume,
                            'high': high,
                            'low': low,
                            'open': open_price
                        })
                    
                    print(f"✅ 成功获取 {len(historical_data)} 条历史数据")
                    return historical_data
            
            raise Exception("API响应格式错误")
            
        except Exception as e:
            print(f"❌ 历史数据获取失败: {e}")
            raise Exception("❌ 无法获取真实历史数据")

class EnhancedPredictionModel:
    """增强版预测模型"""
    
    def __init__(self):
        self.model_30min = None
        self.model_1hour = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy_30min = 0.0
        self.accuracy_1hour = 0.0
        self.mae_30min = 0.0
        self.mae_1hour = 0.0
        self.rmse_30min = 0.0
        self.rmse_1hour = 0.0
        self.feature_names = []
        self.training_data_count = 0
        self.training_time = None
        self.training_features = None
        self.training_targets_30min = None
        self.training_targets_1hour = None
        self.test_targets_30min = None
        self.test_predictions_30min = None
        self.test_targets_1hour = None
        self.test_predictions_1hour = None
        self.direction_accuracy_30min = 0.0
        self.direction_accuracy_1hour = 0.0
        
        # 新增：最佳算法和结果记录
        self.best_algorithm_30min = None
        self.best_algorithm_1hour = None
        self.algorithm_results_30min = {}
        self.algorithm_results_1hour = {}
        
    def prepare_features(self, historical_data):
        """准备特征数据 - 深度优化版特征工程"""
        if len(historical_data) < 100:  # 提高最小数据要求
            raise Exception("历史数据不足，需要至少100条数据进行深度分析")
        
        df = pd.DataFrame(historical_data)
        
        # 确保时间戳格式统一
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            base_time = get_beijing_time().replace(tzinfo=None) - timedelta(hours=len(df))
            df['timestamp'] = [base_time + timedelta(hours=i) for i in range(len(df))]
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. 基础价格特征 - 增强版
        df['price_log'] = np.log(df['price'])
        df['price_diff'] = df['price'].diff()
        df['price_pct_change'] = df['price'].pct_change()
        df['price_log_return'] = df['price_log'].diff()
        
        # 价格变化加速度
        df['price_acceleration'] = df['price_pct_change'].diff()
        df['price_velocity'] = df['price_diff'].rolling(window=3).mean()
        
        # 2. 多层次移动平均线系统
        ma_windows = [3, 5, 8, 10, 13, 20, 21, 30, 50]
        for window in ma_windows:
            if len(df) > window:
                df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
                df[f'price_ma_{window}_ratio'] = df['price'] / df[f'ma_{window}']
                df[f'ma_{window}_slope'] = df[f'ma_{window}'].diff()
                df[f'ma_{window}_std'] = df['price'].rolling(window=window).std()
                
                # 价格相对于移动平均线的偏离度
                df[f'price_ma_{window}_deviation'] = (df['price'] - df[f'ma_{window}']) / df[f'ma_{window}_std']
        
        # 移动平均线交叉信号
        if len(df) > 20:
            df['ma_cross_5_10'] = (df['ma_5'] > df['ma_10']).astype(int)
            df['ma_cross_10_20'] = (df['ma_10'] > df['ma_20']).astype(int)
            df['ma_cross_20_50'] = (df['ma_20'] > df['ma_50']).astype(int) if len(df) > 50 else 0
        
        # 3. 高级技术指标
        # RSI - 多周期
        for period in [6, 14, 21]:
            if len(df) > period:
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-10)  # 避免除零
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD系统 - 多参数
        for fast, slow, signal in [(8, 21, 5), (12, 26, 9), (5, 35, 5)]:
            if len(df) > slow:
                exp_fast = df['price'].ewm(span=fast).mean()
                exp_slow = df['price'].ewm(span=slow).mean()
                macd = exp_fast - exp_slow
                macd_signal = macd.ewm(span=signal).mean()
                macd_histogram = macd - macd_signal
                
                df[f'macd_{fast}_{slow}'] = macd
                df[f'macd_signal_{fast}_{slow}'] = macd_signal
                df[f'macd_histogram_{fast}_{slow}'] = macd_histogram
        
        # 4. 布林带系统 - 多参数
        for period, std_dev in [(10, 1.5), (20, 2), (20, 2.5)]:
            if len(df) > period:
                bb_middle = df['price'].rolling(window=period).mean()
                bb_std = df['price'].rolling(window=period).std()
                bb_upper = bb_middle + (bb_std * std_dev)
                bb_lower = bb_middle - (bb_std * std_dev)
                
                df[f'bb_position_{period}_{std_dev}'] = (df['price'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
                df[f'bb_width_{period}_{std_dev}'] = (bb_upper - bb_lower) / bb_middle
                df[f'bb_squeeze_{period}_{std_dev}'] = (df[f'bb_width_{period}_{std_dev}'] < 
                                                       df[f'bb_width_{period}_{std_dev}'].rolling(20).mean()).astype(int)
        
        # 5. 波动率指标 - 增强版
        for window in [5, 10, 20, 30]:
            if len(df) > window:
                df[f'volatility_{window}'] = df['price_pct_change'].rolling(window=window).std()
                df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(window*2).mean()
                
                # ATR（真实波动范围）
                df[f'tr_{window}'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['price'].shift(1)),
                        abs(df['low'] - df['price'].shift(1))
                    )
                )
                df[f'atr_{window}'] = df[f'tr_{window}'].rolling(window=window).mean()
        
        # 6. 动量指标系统
        for period in [3, 5, 10, 14, 20]:
            if len(df) > period:
                df[f'momentum_{period}'] = df['price'] / df['price'].shift(period) - 1
                df[f'roc_{period}'] = df['price'].pct_change(periods=period)
                
                # Williams %R
                if 'high' in df.columns and 'low' in df.columns:
                    highest_high = df['high'].rolling(window=period).max()
                    lowest_low = df['low'].rolling(window=period).min()
                    df[f'williams_r_{period}'] = (highest_high - df['price']) / (highest_high - lowest_low + 1e-10) * -100
        
        # 7. 成交量分析（增强版）
        if 'volume' in df.columns and df['volume'].sum() > 0:
            # 成交量移动平均和比率
            for window in [5, 10, 20]:
                df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / (df[f'volume_ma_{window}'] + 1e-10)
            
            # 成交量价格趋势（VPT）
            df['vpt'] = (df['volume'] * df['price_pct_change']).cumsum()
            df['vpt_ma'] = df['vpt'].rolling(window=10).mean()
            
            # 资金流指标
            df['money_flow'] = df['price'] * df['volume']
            df['money_flow_ma'] = df['money_flow'].rolling(window=10).mean()
        else:
            # 基于价格波动创建虚拟成交量指标
            df['volume_proxy'] = abs(df['price_pct_change'].fillna(0)) * 1000000
            for window in [5, 10, 20]:
                df[f'volume_ma_{window}'] = df['volume_proxy'].rolling(window=window).mean()
                df[f'volume_ratio_{window}'] = df['volume_proxy'] / (df[f'volume_ma_{window}'] + 1e-10)
        
        # 8. 高级时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_trading_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
        
        # 周期性特征
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 9. 价格位置和排名特征
        for window in [5, 10, 20, 50]:
            if len(df) > window:
                df[f'price_rank_{window}'] = df['price'].rolling(window=window).rank(pct=True)
                df[f'price_quantile_{window}'] = df['price'].rolling(window=window).apply(
                    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
                )
        
        # 10. 趋势强度和方向
        for window in [3, 5, 10, 20]:
            if len(df) > window:
                # 线性回归斜率作为趋势强度
                df[f'trend_slope_{window}'] = df['price'].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )
                
                # 趋势持续性
                df[f'trend_consistency_{window}'] = df['price'].rolling(window=window).apply(
                    lambda x: 1 if len(x) > 1 and x.iloc[-1] > x.iloc[0] else 
                             -1 if len(x) > 1 and x.iloc[-1] < x.iloc[0] else 0
                )
        
        # 11. 支撑阻力位特征
        if len(df) > 20:
            # 局部最高点和最低点
            df['local_max'] = df['price'].rolling(window=5, center=True).max() == df['price']
            df['local_min'] = df['price'].rolling(window=5, center=True).min() == df['price']
            
            # 距离最近支撑阻力位的距离
            recent_max = df['price'].rolling(window=20).max()
            recent_min = df['price'].rolling(window=20).min()
            df['distance_to_resistance'] = (recent_max - df['price']) / recent_max
            df['distance_to_support'] = (df['price'] - recent_min) / recent_min
        
        # 12. 市场情绪指标
        # 恐惧贪婪指数（基于技术指标合成）
        if len(df) > 20:
            fear_greed_components = []
            if 'rsi_14' in df.columns:
                fear_greed_components.append((df['rsi_14'] - 50) / 50)  # RSI偏离
            if 'bb_position_20_2' in df.columns:
                fear_greed_components.append((df['bb_position_20_2'] - 0.5) * 2)  # 布林带位置
            if 'volatility_20' in df.columns:
                vol_norm = (df['volatility_20'] - df['volatility_20'].rolling(50).mean()) / df['volatility_20'].rolling(50).std()
                fear_greed_components.append(-vol_norm.fillna(0))  # 波动率（反向）
            
            if fear_greed_components:
                df['market_sentiment'] = np.mean(fear_greed_components, axis=0)
        
        # 选择最有效的特征
        feature_columns = []
        
        # 基础特征
        basic_features = ['price_log', 'price_pct_change', 'price_log_return', 'price_acceleration', 'price_velocity']
        feature_columns.extend([f for f in basic_features if f in df.columns])
        
        # 移动平均线特征
        ma_features = [col for col in df.columns if 'ma_' in col and ('ratio' in col or 'slope' in col or 'deviation' in col)]
        feature_columns.extend(ma_features)
        
        # 交叉信号
        cross_features = [col for col in df.columns if 'cross' in col]
        feature_columns.extend(cross_features)
        
        # 技术指标
        tech_features = [col for col in df.columns if any(indicator in col for indicator in ['rsi_', 'macd_', 'bb_', 'williams_r_'])]
        feature_columns.extend(tech_features)
        
        # 波动率和动量
        vol_mom_features = [col for col in df.columns if any(indicator in col for indicator in ['volatility_', 'atr_', 'momentum_', 'roc_'])]
        feature_columns.extend(vol_mom_features)
        
        # 成交量特征
        volume_features = [col for col in df.columns if 'volume' in col or 'money_flow' in col or 'vpt' in col]
        feature_columns.extend(volume_features)
        
        # 时间特征
        time_features = ['hour', 'day_of_week', 'is_weekend', 'is_trading_hours', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        feature_columns.extend([f for f in time_features if f in df.columns])
        
        # 价格位置特征
        position_features = [col for col in df.columns if 'rank_' in col or 'quantile_' in col]
        feature_columns.extend(position_features)
        
        # 趋势特征
        trend_features = [col for col in df.columns if 'trend_' in col or 'distance_to_' in col]
        feature_columns.extend(trend_features)
        
        # 市场情绪
        if 'market_sentiment' in df.columns:
            feature_columns.append('market_sentiment')
        
        # 去重并过滤有效特征
        feature_columns = list(set(feature_columns))
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # 移除包含过多NaN的特征
        valid_features = []
        for col in feature_columns:
            if df[col].notna().sum() / len(df) > 0.7:  # 至少70%的数据有效
                valid_features.append(col)
        
        feature_columns = valid_features
        
        # 移除包含NaN的行
        features_df = df[feature_columns].dropna()
        
        # 数值稳定性处理
        print(f"🔧 进行数值稳定性处理...")
        
        # 替换无穷大值
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # 填充剩余的NaN值
        for col in features_df.columns:
            if features_df[col].isna().any():
                # 使用中位数填充数值型特征的NaN
                median_val = features_df[col].median()
                if pd.isna(median_val):
                    features_df[col] = features_df[col].fillna(0)
                else:
                    features_df[col] = features_df[col].fillna(median_val)
        
        # 检查异常值并进行裁剪
        for col in features_df.columns:
            Q1 = features_df[col].quantile(0.01)
            Q99 = features_df[col].quantile(0.99)
            features_df[col] = features_df[col].clip(lower=Q1, upper=Q99)
        
        # 最终验证
        problematic_cols = []
        for col in features_df.columns:
            if not np.isfinite(features_df[col]).all():
                problematic_cols.append(col)
        
        if problematic_cols:
            print(f"⚠️ 移除有问题的特征: {problematic_cols}")
            features_df = features_df.drop(columns=problematic_cols)
            feature_columns = [col for col in feature_columns if col not in problematic_cols]
        
        # 保存特征名称
        self.feature_names = feature_columns
        
        print(f"🔧 特征工程完成: {len(feature_columns)} 个特征, {len(features_df)} 条有效数据")
        
        # 返回特征、价格和时间戳
        valid_indices = features_df.index
        prices = df.loc[valid_indices, 'price'].values
        timestamps = df.loc[valid_indices, 'timestamp'].values
        
        return features_df, prices, timestamps
    
    def create_targets(self, prices, timestamps):
        """创建目标变量 - 二分类版本（只有上涨/下跌）"""
        targets_30min = []
        targets_1hour = []
        
        print("🎯 创建智能目标变量（二分类：上涨/下跌）...")
        
        # 将时间戳转换为pandas时间序列以便处理
        timestamps_pd = pd.to_datetime(timestamps)
        
        for i in range(len(prices)):
            current_time = timestamps_pd[i]
            current_price = prices[i]
            
            # 30分钟目标 - 预测短期趋势（更敏感）
            target_30min = np.nan
            # 寻找最接近30分钟后的价格，如果找不到就用下一个点
            if i + 1 < len(prices):
                # 对于30分钟预测，使用下一个时间点，但增加更严格的判断
                next_price = prices[i + 1]
                price_change_pct = (next_price - current_price) / current_price
                
                # 30分钟模型：使用较小的阈值，更敏感
                threshold_30min = 0.001  # 0.1%阈值
                if price_change_pct > threshold_30min:
                    target_30min = 1  # 上涨
                elif price_change_pct < -threshold_30min:
                    target_30min = -1  # 下跌
                else:
                    # 对于微小变化，基于历史趋势判断
                    if i >= 2:
                        recent_trend = (current_price - prices[i-2]) / prices[i-2]
                        if recent_trend > 0:
                            target_30min = 1
                        else:
                            target_30min = -1
                    else:
                        target_30min = 1 if price_change_pct >= 0 else -1
            
            # 1小时目标 - 预测中期趋势（更稳定）
            target_1hour = np.nan
            # 寻找2个时间点后的价格，模拟真正的1小时后
            if i + 2 < len(prices):
                future_price = prices[i + 2]
                price_change_pct = (future_price - current_price) / current_price
                
                # 1小时模型：使用较大的阈值，更稳定
                threshold_1hour = 0.005  # 0.5%阈值
                if price_change_pct > threshold_1hour:
                    target_1hour = 1  # 上涨
                elif price_change_pct < -threshold_1hour:
                    target_1hour = -1  # 下跌
                else:
                    # 对于中等变化，考虑更长的历史趋势
                    if i >= 5:
                        long_trend = (current_price - prices[i-5]) / prices[i-5]
                        if long_trend > 0.002:  # 0.2%以上认为是上涨趋势
                            target_1hour = 1
                        elif long_trend < -0.002:
                            target_1hour = -1
                        else:
                            target_1hour = 1 if price_change_pct >= 0 else -1
                    else:
                        target_1hour = 1 if price_change_pct >= 0 else -1
            elif i + 1 < len(prices):
                # 如果找不到2个点后的价格，用下一个点但更保守
                next_price = prices[i + 1]
                price_change_pct = (next_price - current_price) / current_price
                # 1小时目标更保守，只在明显变化时才确定方向
                if abs(price_change_pct) > 0.003:
                    target_1hour = 1 if price_change_pct > 0 else -1
                else:
                    # 保守策略：微小变化时基于长期趋势
                    if i >= 3:
                        trend = (current_price - prices[i-3]) / prices[i-3]
                        target_1hour = 1 if trend > 0 else -1
                    else:
                        target_1hour = 1 if price_change_pct >= 0 else -1
            
            targets_30min.append(target_30min)
            targets_1hour.append(target_1hour)
        
        targets_30min = np.array(targets_30min)
        targets_1hour = np.array(targets_1hour)
        
        # 打印目标分布情况
        valid_30 = ~pd.isna(targets_30min)
        valid_1h = ~pd.isna(targets_1hour)
        
        if valid_30.sum() > 0:
            up_30 = (targets_30min[valid_30] == 1).sum()
            down_30 = (targets_30min[valid_30] == -1).sum()
            print(f"   30分钟目标分布: 上涨{up_30}, 下跌{down_30}")
        
        if valid_1h.sum() > 0:
            up_1h = (targets_1hour[valid_1h] == 1).sum()
            down_1h = (targets_1hour[valid_1h] == -1).sum()
            print(f"   1小时目标分布: 上涨{up_1h}, 下跌{down_1h}")
        
        return targets_30min, targets_1hour
    
    def train_models(self, historical_data):
        """训练预测模型 - 多算法优化版 + 性能优化"""
        print("🧠 开始训练多算法优化版预测模型...")
        self.training_time = get_beijing_time()  # 使用北京时间
        
        try:
            # ⚡ 性能优化：缓存数据处理结果
            data_hash = hash(str(len(historical_data)) + str(historical_data[0] if historical_data else ""))
            
            with st.spinner("🔄 正在处理训练数据..."):
                # 准备数据
                features, prices, timestamps = self.prepare_features(historical_data)
                self.training_data_count = len(features)
                
                # 创建目标变量
                targets_30min, targets_1hour = self.create_targets(prices, timestamps)
                
                # 移除NaN值
                valid_indices = ~(np.isnan(targets_30min) | np.isnan(targets_1hour))
                features_clean = features[valid_indices]
                targets_30min_clean = targets_30min[valid_indices]
                targets_1hour_clean = targets_1hour[valid_indices]
                
                if len(features_clean) < 50:
                    raise Exception(f"❌ 有效训练数据不足: 需要至少50条，当前只有{len(features_clean)}条")
                
                # 保存训练数据用于展示
                self.training_features = features_clean.copy()
                self.training_targets_30min = targets_30min_clean.copy()
                self.training_targets_1hour = targets_1hour_clean.copy()
            
            # 标准化特征
            with st.spinner("🔧 正在标准化特征..."):
                features_scaled = self.scaler.fit_transform(features_clean)
                
                # 将-1/1类别转换为0/1以适应XGBoost等算法
                targets_30min_binary = np.where(targets_30min_clean == -1, 0, 1)
                targets_1hour_binary = np.where(targets_1hour_clean == -1, 0, 1)
                
                # 使用时间序列分割
                split_point = int(len(features_scaled) * 0.8)
                X_train, X_test = features_scaled[:split_point], features_scaled[split_point:]
                y_train_30, y_test_30 = targets_30min_binary[:split_point], targets_30min_binary[split_point:]
                y_train_1h, y_test_1h = targets_1hour_binary[:split_point], targets_1hour_binary[split_point:]
            
            print("🔍 开始多算法评估...")
            
            # ⚡ 优化：减少算法数量，选择最优的几个
            from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            
            algorithms = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=200,  # 减少树的数量
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'XGBoost': xgb.XGBClassifier(
                    n_estimators=200,  # 减少迭代次数
                    max_depth=6,
                    learning_rate=0.1,  # 提高学习率
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                    use_label_encoder=False,
                    eval_metric='logloss'
                ),
                'LightGBM': lgb.LGBMClassifier(
                    n_estimators=200,  # 减少迭代次数
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                    class_weight='balanced'
                ),
                'ExtraTrees': ExtraTreesClassifier(
                    n_estimators=150,  # 减少树的数量
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100,  # 减少迭代次数
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'LogisticRegression': LogisticRegression(
                    C=1.0,
                    max_iter=500,  # 减少迭代次数
                    random_state=42,
                    class_weight='balanced',
                    solver='liblinear'
                )
            }
            
            # 评估所有算法 - 30分钟模型
            with st.spinner("📊 训练30分钟预测模型..."):
                print("📊 评估30分钟预测模型...")
                best_score_30 = -float('inf')
                best_model_30 = None
                best_name_30 = ""
                results_30 = {}
                
                for name, model in algorithms.items():
                    try:
                        print(f"   测试 {name}...")
                        
                        # 训练模型
                        model.fit(X_train, y_train_30)
                        
                        # 预测
                        y_pred_30 = model.predict(X_test)
                        
                        # 计算准确率 - 二分类
                        accuracy = np.mean(y_pred_30 == y_test_30)
                        
                        # 计算各类别的准确率
                        from sklearn.metrics import f1_score
                        
                        # 方向预测准确率（二分类：上涨/下跌直接对比）
                        direction_accuracy = accuracy  # 二分类情况下，总准确率就是方向准确率
                        
                        # F1分数（二分类）
                        try:
                            f1 = f1_score(y_test_30, y_pred_30, average='binary', pos_label=1)
                        except:
                            f1 = 0.3
                        
                        # 综合评分 - 重点关注方向预测准确率
                        composite_score = direction_accuracy * 0.8 + f1 * 0.2
                        
                        results_30[name] = {
                            'model': model,
                            'accuracy': accuracy,
                            'direction_accuracy': direction_accuracy,
                            'f1_score': f1,
                            'composite_score': composite_score,
                            'predictions': y_pred_30
                        }
                        
                        print(f"      总准确率: {accuracy:.1%}, 方向准确率: {direction_accuracy:.1%}, F1: {f1:.3f}, 综合评分: {composite_score:.3f}")
                        
                        if composite_score > best_score_30:
                            best_score_30 = composite_score
                            best_model_30 = model
                            best_name_30 = name
                        
                    except Exception as e:
                        print(f"      ❌ {name} 训练失败: {e}")
                        continue
            
            # 评估所有算法 - 1小时模型
            print("📊 评估1小时预测模型...")
            best_score_1h = -float('inf')
            best_model_1h = None
            best_name_1h = ""
            results_1h = {}
            
            for name, model in algorithms.items():
                try:
                    print(f"   测试 {name}...")
                    
                    # 重新创建新的算法实例，避免使用30分钟模型的训练过的实例
                    if name == 'RandomForest':
                        model = RandomForestClassifier(
                            n_estimators=200, max_depth=10, min_samples_split=5,
                            min_samples_leaf=2, max_features='sqrt', bootstrap=True,
                            random_state=42, n_jobs=-1, class_weight='balanced'
                        )
                    elif name == 'XGBoost':
                        model = xgb.XGBClassifier(
                            n_estimators=200, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                            reg_alpha=0.1, reg_lambda=1, random_state=42, n_jobs=-1,
                            verbosity=0, use_label_encoder=False, eval_metric='logloss',
                            objective='binary:logistic'
                        )
                    elif name == 'LightGBM':
                        model = lgb.LGBMClassifier(
                            n_estimators=200, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                            reg_lambda=1, random_state=42, n_jobs=-1, verbosity=-1,
                            class_weight='balanced', objective='binary'
                        )
                    elif name == 'ExtraTrees':
                        model = ExtraTreesClassifier(
                            n_estimators=150, max_depth=10, min_samples_split=5,
                            random_state=42, n_jobs=-1, class_weight='balanced'
                        )
                    elif name == 'GradientBoosting':
                        model = GradientBoostingClassifier(
                            n_estimators=100, learning_rate=0.1, max_depth=5,
                            random_state=42
                        )
                    elif name == 'LogisticRegression':
                        model = LogisticRegression(
                            C=1.0, max_iter=500, random_state=42,
                            class_weight='balanced', solver='liblinear'
                        )
                    else:
                        continue
                    
                    # 训练模型
                    model.fit(X_train, y_train_1h)
                    
                    # 预测
                    y_pred_1h = model.predict(X_test)
                    
                    # 计算准确率 - 二分类
                    accuracy = np.mean(y_pred_1h == y_test_1h)
                    
                    # 方向预测准确率（二分类：上涨/下跌直接对比）
                    direction_accuracy = accuracy  # 二分类情况下，总准确率就是方向准确率
                    
                    # F1分数（二分类）
                    try:
                        f1 = f1_score(y_test_1h, y_pred_1h, average='binary', pos_label=1)
                    except:
                        f1 = 0.3
                    
                    # 综合评分
                    composite_score = direction_accuracy * 0.8 + f1 * 0.2
                    
                    results_1h[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'direction_accuracy': direction_accuracy,
                        'f1_score': f1,
                        'composite_score': composite_score,
                        'predictions': y_pred_1h
                    }
                    
                    print(f"      总准确率: {accuracy:.1%}, 方向准确率: {direction_accuracy:.1%}, F1: {f1:.3f}, 综合评分: {composite_score:.3f}")
                    
                    if composite_score > best_score_1h:
                        best_score_1h = composite_score
                        best_model_1h = model
                        best_name_1h = name
                        
                except Exception as e:
                    print(f"      ❌ {name} 训练失败: {e}")
                    continue
            
            # 保存最佳模型
            if best_model_30 is not None:
                self.model_30min = best_model_30
                self.accuracy_30min = results_30[best_name_30]['accuracy']
                self.direction_accuracy_30min = results_30[best_name_30]['direction_accuracy']
                self.mae_30min = results_30[best_name_30]['f1_score']  # 现在存储F1分数
                self.rmse_30min = results_30[best_name_30]['composite_score']  # 存储综合评分
                self.test_targets_30min = y_test_30
                self.test_predictions_30min = results_30[best_name_30]['predictions']
                self.best_algorithm_30min = best_name_30
                
                print(f"✅ 30分钟最佳模型: {best_name_30} (趋势准确率: {self.direction_accuracy_30min:.1%})")
            else:
                raise Exception("❌ 无法找到有效的30分钟预测模型")
            
            if best_model_1h is not None:
                self.model_1hour = best_model_1h
                self.accuracy_1hour = results_1h[best_name_1h]['accuracy']
                self.direction_accuracy_1hour = results_1h[best_name_1h]['direction_accuracy']
                self.mae_1hour = results_1h[best_name_1h]['f1_score']  # 现在存储F1分数
                self.rmse_1hour = results_1h[best_name_1h]['composite_score']  # 存储综合评分
                self.test_targets_1hour = y_test_1h
                self.test_predictions_1hour = results_1h[best_name_1h]['predictions']
                self.best_algorithm_1hour = best_name_1h
                
                print(f"✅ 1小时最佳模型: {best_name_1h} (趋势准确率: {self.direction_accuracy_1hour:.1%})")
            else:
                raise Exception("❌ 无法找到有效的1小时预测模型")
            
            # 保存所有算法结果用于分析
            self.algorithm_results_30min = results_30
            self.algorithm_results_1hour = results_1h
            
            self.is_trained = True
            
            print(f"🎉 多算法优化完成!")
            print(f"📊 训练数据量: {self.training_data_count} 条真实数据")
            print(f"🏆 30分钟最佳: {best_name_30} - 方向准确率: {self.direction_accuracy_30min:.1%}, F1: {self.mae_30min:.6f}")
            print(f"🏆 1小时最佳: {best_name_1h} - 方向准确率: {self.direction_accuracy_1hour:.1%}, F1: {self.mae_1hour:.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, current_features):
        """进行预测 - 二分类版本"""
        if not self.is_trained:
            return None
        
        try:
            features_scaled = self.scaler.transform([current_features])
            
            # 获取分类预测和概率
            pred_class_30min_raw = self.model_30min.predict(features_scaled)[0]
            pred_class_1hour_raw = self.model_1hour.predict(features_scaled)[0]
            
            # 转换0/1预测结果回-1/1格式
            pred_class_30min = -1 if pred_class_30min_raw == 0 else 1
            pred_class_1hour = -1 if pred_class_1hour_raw == 0 else 1
            
            # 获取预测概率 - 二分类版本
            if hasattr(self.model_30min, 'predict_proba'):
                prob_30min_raw = self.model_30min.predict_proba(features_scaled)[0]
                print(f"30分钟原始概率: {prob_30min_raw}")
                
                # 二分类：[下跌概率(0类), 上涨概率(1类)]
                if len(prob_30min_raw) == 2:
                    down_prob_30 = prob_30min_raw[0]  # 下跌（0类 -> -1类）概率
                    up_prob_30 = prob_30min_raw[1]    # 上涨（1类）概率
                else:
                    # 异常情况
                    down_prob_30, up_prob_30 = 0.5, 0.5
            else:
                # 模型不支持概率预测，根据类别生成概率
                if pred_class_30min == 1:
                    down_prob_30, up_prob_30 = 0.2, 0.8
                else:
                    down_prob_30, up_prob_30 = 0.8, 0.2
            
            # 1小时模型概率处理
            if hasattr(self.model_1hour, 'predict_proba'):
                prob_1hour_raw = self.model_1hour.predict_proba(features_scaled)[0]
                print(f"1小时原始概率: {prob_1hour_raw}")
                
                if len(prob_1hour_raw) == 2:
                    down_prob_1h = prob_1hour_raw[0]  # 下跌（0类 -> -1类）概率
                    up_prob_1h = prob_1hour_raw[1]    # 上涨（1类）概率
                else:
                    down_prob_1h, up_prob_1h = 0.5, 0.5
            else:
                if pred_class_1hour == 1:
                    down_prob_1h, up_prob_1h = 0.2, 0.8
                else:
                    down_prob_1h, up_prob_1h = 0.8, 0.2
            
            # 检查并修复异常概率值
            def fix_extreme_probabilities(up_prob, down_prob, threshold=0.95):
                """修复极端概率值，避免100%或0%的情况"""
                # 确保概率和为1
                total = up_prob + down_prob
                if total > 0:
                    up_prob = up_prob / total
                    down_prob = down_prob / total
                
                # 如果某个概率过高，进行平滑处理
                if up_prob > threshold:
                    up_prob = threshold
                    down_prob = 1.0 - threshold
                elif down_prob > threshold:
                    down_prob = threshold
                    up_prob = 1.0 - threshold
                
                # 确保最小概率不低于5%
                min_prob = 0.05
                if up_prob < min_prob:
                    up_prob = min_prob
                    down_prob = 1.0 - min_prob
                elif down_prob < min_prob:
                    down_prob = min_prob
                    up_prob = 1.0 - min_prob
                
                return up_prob, down_prob
            
            # 应用概率修复
            up_prob_30, down_prob_30 = fix_extreme_probabilities(up_prob_30, down_prob_30)
            up_prob_1h, down_prob_1h = fix_extreme_probabilities(up_prob_1h, down_prob_1h)
            
            print(f"修复后30分钟概率: 下跌{down_prob_30:.3f}, 上涨{up_prob_30:.3f}")
            print(f"修复后1小时概率: 下跌{down_prob_1h:.3f}, 上涨{up_prob_1h:.3f}")
            
            return {
                '30min': {
                    'class': int(pred_class_30min),
                    'up_prob': up_prob_30,
                    'down_prob': down_prob_30
                },
                '1hour': {
                    'class': int(pred_class_1hour),
                    'up_prob': up_prob_1h,
                    'down_prob': down_prob_1h
                }
            }
        except Exception as e:
            print(f"预测错误: {e}")
            return None
    
    def generate_trading_signal(self, predictions, current_price):
        """生成明确的AI交易信号 - 二分类版本"""
        if not predictions:
            return {
                'action': '❓ 无法分析',
                'confidence': '低',
                'reason': '模型预测失败',
                'signal_strength': 0,
                'color': '#888888'
            }
        
        pred_30min = predictions['30min']
        pred_1hour = predictions['1hour']
        
        # 获取预测类别和概率
        class_30 = pred_30min['class']
        class_1h = pred_1hour['class']
        
        up_prob_30 = pred_30min['up_prob'] * 100
        down_prob_30 = pred_30min['down_prob'] * 100
        up_prob_1h = pred_1hour['up_prob'] * 100
        down_prob_1h = pred_1hour['down_prob'] * 100
        
        # 综合评判
        avg_up_prob = (up_prob_30 + up_prob_1h) / 2
        avg_down_prob = (down_prob_30 + down_prob_1h) / 2
        
        # 方向一致性检查（二分类）
        direction_consistent = (class_30 == class_1h)
        
        # 预测强度（基于概率差异）
        strength_30 = abs(up_prob_30 - down_prob_30)
        strength_1h = abs(up_prob_1h - down_prob_1h)
        avg_strength = (strength_30 + strength_1h) / 2
        
        # 决策逻辑
        signal = {}
        
        # 置信度阈值
        HIGH_CONFIDENCE_THRESHOLD = 70
        MEDIUM_CONFIDENCE_THRESHOLD = 60
        CONSISTENCY_BONUS = 10  # 方向一致性加分
        
        # 调整置信度（如果方向一致，增加置信度）
        if direction_consistent:
            max_prob = max(avg_up_prob, avg_down_prob) + CONSISTENCY_BONUS
            signal_strength = min(100, avg_strength + CONSISTENCY_BONUS)
        else:
            max_prob = max(avg_up_prob, avg_down_prob)
            signal_strength = avg_strength
        
        # 生成具体信号
        if max_prob >= HIGH_CONFIDENCE_THRESHOLD and direction_consistent:
            # 高置信度信号
            if avg_up_prob > avg_down_prob:
                price_change_estimate = (avg_up_prob / 100) * 0.02  # 估算2%的变化
                target_price = current_price * (1 + price_change_estimate)
                signal = {
                    'action': '🚀 强烈建议买涨',
                    'confidence': '高',
                    'reason': f'双时段一致看涨，平均上涨概率{avg_up_prob:.1f}%',
                    'signal_strength': int(signal_strength),
                    'color': '#00C851',
                    'priority': '🔥 优先级：高',
                    'target_price': target_price
                }
            else:
                price_change_estimate = (avg_down_prob / 100) * 0.02
                target_price = current_price * (1 - price_change_estimate)
                signal = {
                    'action': '📉 强烈建议买跌',
                    'confidence': '高',
                    'reason': f'双时段一致看跌，平均下跌概率{avg_down_prob:.1f}%',
                    'signal_strength': int(signal_strength),
                    'color': '#FF4444',
                    'priority': '🔥 优先级：高',
                    'target_price': target_price
                }
        
        elif max_prob >= MEDIUM_CONFIDENCE_THRESHOLD:
            # 中等置信度信号
            if avg_up_prob > avg_down_prob:
                consistency_note = "方向一致" if direction_consistent else "存在分歧"
                price_change_estimate = (avg_up_prob / 100) * 0.01
                target_price = current_price * (1 + price_change_estimate)
                signal = {
                    'action': '📈 建议买涨',
                    'confidence': '中',
                    'reason': f'上涨概率{avg_up_prob:.1f}%，{consistency_note}',
                    'signal_strength': int(signal_strength),
                    'color': '#28A745',
                    'priority': '⚡ 优先级：中',
                    'target_price': target_price
                }
            else:
                consistency_note = "方向一致" if direction_consistent else "存在分歧"
                price_change_estimate = (avg_down_prob / 100) * 0.01
                target_price = current_price * (1 - price_change_estimate)
                signal = {
                    'action': '📉 建议买跌',
                    'confidence': '中',
                    'reason': f'下跌概率{avg_down_prob:.1f}%，{consistency_note}',
                    'signal_strength': int(signal_strength),
                    'color': '#DC3545',
                    'priority': '⚡ 优先级：中',
                    'target_price': target_price
                }
        
        else:
            # 低置信度或不确定信号
            if not direction_consistent:
                signal = {
                    'action': '⚠️ 暂时观望',
                    'confidence': '低',
                    'reason': f'双时段预测方向不一致，建议等待更明确信号',
                    'signal_strength': int(signal_strength),
                    'color': '#FFC107',
                    'priority': '💤 优先级：低',
                    'target_price': current_price
                }
            else:
                signal = {
                    'action': '🤔 谨慎观望',
                    'confidence': '低',
                    'reason': f'预测置信度不足({max_prob:.1f}%)，建议等待',
                    'signal_strength': int(signal_strength),
                    'color': '#6C757D',
                    'priority': '💤 优先级：低',
                    'target_price': current_price
                }
        
        # 添加额外信息
        signal['details'] = {
            '30min_prediction': f"{'📈' if class_30 > 0 else '📉'} {class_30}类({up_prob_30:.1f}%↑)",
            '1hour_prediction': f"{'📈' if class_1h > 0 else '📉'} {class_1h}类({up_prob_1h:.1f}%↑)",
            'direction_consistent': direction_consistent,
            'model_accuracy_30min': f"{self.direction_accuracy_30min:.1%}",
            'model_accuracy_1hour': f"{self.direction_accuracy_1hour:.1%}"
        }
        
        return signal

def show_training_data_page():
    """显示训练数据页面"""
    st.markdown("## 📊 训练数据分析")
    
    if 'enh_prediction_model' not in st.session_state or not st.session_state.enh_prediction_model.is_trained:
        st.warning("⚠️ 请先训练模型以查看训练数据")
        return
    
    model = st.session_state.enh_prediction_model
    
    # 训练信息概览
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 训练数据量", f"{model.training_data_count} 条")
    with col2:
        training_time_str = format_beijing_time(model.training_time) if model.training_time else "未知"
        st.metric("🕒 训练时间", training_time_str)
    with col3:
        st.metric("📈 特征数量", len(model.feature_names))
    with col4:
        st.metric("🎯 目标变量", "2个 (30分钟+1小时)")
    
    # 模型性能指标
    st.markdown("### 🎯 模型性能指标")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm_30 = getattr(model, 'best_algorithm_30min', '未知')
        st.markdown(f"""
        <div class="training-card">
            <h4>📊 30分钟预测模型</h4>
            <p><strong>最佳算法:</strong> {algorithm_30}</p>
            <p><strong>总体准确率:</strong> {model.accuracy_30min:.1%}</p>
            <p><strong>趋势准确率:</strong> {model.direction_accuracy_30min:.1%}</p>
            <p><strong>F1分数:</strong> {model.mae_30min:.3f}</p>
            <p><strong>综合评分:</strong> {model.rmse_30min:.3f}</p>
            <p><strong>测试集样本数:</strong> {len(model.test_targets_30min) if hasattr(model, 'test_targets_30min') and model.test_targets_30min is not None else 0} 个</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        algorithm_1h = getattr(model, 'best_algorithm_1hour', '未知')
        st.markdown(f"""
        <div class="training-card">
            <h4>📊 1小时预测模型</h4>
            <p><strong>最佳算法:</strong> {algorithm_1h}</p>
            <p><strong>总体准确率:</strong> {model.accuracy_1hour:.1%}</p>
            <p><strong>趋势准确率:</strong> {model.direction_accuracy_1hour:.1%}</p>
            <p><strong>F1分数:</strong> {model.mae_1hour:.3f}</p>
            <p><strong>综合评分:</strong> {model.rmse_1hour:.3f}</p>
            <p><strong>测试集样本数:</strong> {len(model.test_targets_1hour) if hasattr(model, 'test_targets_1hour') and model.test_targets_1hour is not None else 0} 个</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 算法对比结果
    if hasattr(model, 'algorithm_results_30min') and model.algorithm_results_30min:
        st.markdown("### 🏆 算法性能对比")
        
        # 创建对比数据
        comparison_data = []
        for name, result in model.algorithm_results_30min.items():
            comparison_data.append({
                '算法': name,
                '30分钟准确率': f"{result['accuracy']:.1%}",
                '30分钟趋势准确率': f"{result['direction_accuracy']:.1%}",
                '30分钟F1分数': f"{result['f1_score']:.3f}",
                '30分钟综合评分': f"{result['composite_score']:.3f}"
            })
        
        # 添加1小时结果
        for i, (name, result) in enumerate(model.algorithm_results_1hour.items()):
            if i < len(comparison_data):
                comparison_data[i]['1小时准确率'] = f"{result['accuracy']:.1%}"
                comparison_data[i]['1小时趋势准确率'] = f"{result['direction_accuracy']:.1%}"
                comparison_data[i]['1小时F1分数'] = f"{result['f1_score']:.3f}"
                comparison_data[i]['1小时综合评分'] = f"{result['composite_score']:.3f}"
        
        # 显示对比表格
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # 可视化对比
        col1, col2 = st.columns(2)
        
        with col1:
            # 30分钟算法对比
            accuracy_30_data = [(name, result['accuracy']) for name, result in model.algorithm_results_30min.items()]
            accuracy_30_data.sort(key=lambda x: x[1], reverse=True)
            
            fig = px.bar(
                x=[item[1] for item in accuracy_30_data],
                y=[item[0] for item in accuracy_30_data],
                orientation='h',
                title="30分钟模型算法准确率对比",
                labels={"x": "方向预测准确率", "y": "算法"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 1小时算法对比
            accuracy_1h_data = [(name, result['accuracy']) for name, result in model.algorithm_results_1hour.items()]
            accuracy_1h_data.sort(key=lambda x: x[1], reverse=True)
            
            fig = px.bar(
                x=[item[1] for item in accuracy_1h_data],
                y=[item[0] for item in accuracy_1h_data],
                orientation='h',
                title="1小时模型算法准确率对比",
                labels={"x": "方向预测准确率", "y": "算法"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # 特征重要性
    if (hasattr(model.model_30min, 'feature_importances_') and 
        hasattr(model.model_1hour, 'feature_importances_')):
        st.markdown("### 📈 特征重要性分析")
        
        # 30分钟模型特征重要性
        importance_30min = model.model_30min.feature_importances_
        importance_1hour = model.model_1hour.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': model.feature_names,
            'importance_30min': importance_30min,
            'importance_1hour': importance_1hour
        }).sort_values('importance_30min', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                importance_df.head(10), 
                x='importance_30min', 
                y='feature',
                title="30分钟模型 - Top 10 特征重要性",
                orientation='h'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                importance_df.head(10), 
                x='importance_1hour', 
                y='feature',
                title="1小时模型 - Top 10 特征重要性",
                orientation='h'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("### 📈 特征重要性分析")
        st.info(f"⚠️ 当前最佳模型 ({getattr(model, 'best_algorithm_30min', '未知')} / {getattr(model, 'best_algorithm_1hour', '未知')}) 不支持特征重要性分析")
        
        # 显示模型类型信息
        st.markdown(f"""
        <div class="metric-card">
            <h5>📊 模型信息</h5>
            <p><strong>30分钟最佳算法:</strong> {getattr(model, 'best_algorithm_30min', '未知')}</p>
            <p><strong>1小时最佳算法:</strong> {getattr(model, 'best_algorithm_1hour', '未知')}</p>
            <p><strong>说明:</strong> GaussianNB、SVC等模型不提供特征重要性信息</p>
            <p><strong>替代方案:</strong> 可使用RandomForest、XGBoost等树模型查看特征重要性</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 目标变量分布
    if hasattr(model, 'training_targets_30min'):
        st.markdown("### 📊 目标变量分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                x=model.training_targets_30min,
                title="30分钟价格变化分布",
                nbins=50
            )
            fig.update_xaxes(title="价格变化率")
            fig.update_yaxes(title="频次")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                x=model.training_targets_1hour,
                title="1小时价格变化分布",
                nbins=50
            )
            fig.update_xaxes(title="价格变化率")
            fig.update_yaxes(title="频次")
            st.plotly_chart(fig, use_container_width=True)

    # 测试集预测结果可视化
    if hasattr(model, 'test_targets_30min') and model.test_targets_30min is not None:
        st.markdown("### 📊 测试集预测效果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 30分钟模型预测效果
            fig = px.scatter(
                x=model.test_targets_30min,
                y=model.test_predictions_30min,
                title="30分钟模型：实际值 vs 预测值",
                labels={"x": "实际价格变化", "y": "预测价格变化"}
            )
            # 添加理想预测线
            min_val = min(model.test_targets_30min.min(), model.test_predictions_30min.min())
            max_val = max(model.test_targets_30min.max(), model.test_predictions_30min.max())
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash"),
                name="理想预测线"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 1小时模型预测效果
            if hasattr(model, 'test_targets_1hour') and model.test_targets_1hour is not None:
                fig = px.scatter(
                    x=model.test_targets_1hour,
                    y=model.test_predictions_1hour,
                    title="1小时模型：实际值 vs 预测值",
                    labels={"x": "实际价格变化", "y": "预测价格变化"}
                )
                # 添加理想预测线
                min_val = min(model.test_targets_1hour.min(), model.test_predictions_1hour.min())
                max_val = max(model.test_targets_1hour.max(), model.test_predictions_1hour.max())
                fig.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash"),
                    name="理想预测线"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # 方向预测准确率可视化
        st.markdown("### 🎯 方向预测准确率分析")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 30分钟方向准确率
            accuracy_30 = model.direction_accuracy_30min * 100
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = accuracy_30,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "30分钟方向准确率"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#00C851" if accuracy_30 >= 60 else "#FFC107" if accuracy_30 >= 50 else "#FF4444"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 1小时方向准确率
            accuracy_1h = model.direction_accuracy_1hour * 100
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = accuracy_1h,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "1小时方向准确率"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#00C851" if accuracy_1h >= 60 else "#FFC107" if accuracy_1h >= 50 else "#FF4444"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # 综合评估
            avg_accuracy = (accuracy_30 + accuracy_1h) / 2
            test_samples = len(model.test_targets_30min)
            
            st.markdown(f"""
            <div class="metric-card">
                <h5>📈 综合评估</h5>
                <p><strong>平均方向准确率:</strong> {avg_accuracy:.1f}%</p>
                <p><strong>测试样本数:</strong> {test_samples} 个</p>
                <p><strong>模型状态:</strong> {'🟢 良好' if avg_accuracy >= 60 else '🟡 一般' if avg_accuracy >= 50 else '🔴 需改进'}</p>
                <p><strong>建议:</strong> {'可信度较高' if avg_accuracy >= 60 else '谨慎参考' if avg_accuracy >= 50 else '仅供参考'}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """主程序"""
    
    # 初始化会话状态
    if 'enh_eth_api' not in st.session_state:
        st.session_state.enh_eth_api = EnhancedETHAPI()
    if 'enh_prediction_model' not in st.session_state:
        st.session_state.enh_prediction_model = EnhancedPredictionModel()
    if 'enh_historical_data' not in st.session_state:
        st.session_state.enh_historical_data = None
    if 'enh_model_trained' not in st.session_state:
        st.session_state.enh_model_trained = False
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = False
    if 'last_training_time' not in st.session_state:
        st.session_state.last_training_time = None
    if 'system_start_time' not in st.session_state:
        st.session_state.system_start_time = get_beijing_time().replace(tzinfo=None)
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # 页面导航
    page = st.sidebar.selectbox(
        "📋 选择页面",
        ["🏠 实时监控", "📊 训练数据分析", "⚙️ 系统设置"]
    )
    
    if page == "📊 训练数据分析":
        show_training_data_page()
        return
    elif page == "⚙️ 系统设置":
        st.markdown("## ⚙️ 系统设置")
        st.info("系统设置功能开发中...")
        return
    
    # 主页面
    st.markdown("""
    <div class="main-header">
        <h1>🚀 增强版ETH监控系统 v8.0.0</h1>
        <p>一个月历史数据 + 训练数据展示 + 自动刷新 + 定时重训练</p>
    </div>
    """, unsafe_allow_html=True)
    
    api = st.session_state.enh_eth_api
    model = st.session_state.enh_prediction_model
    
    # 侧边栏控制
    with st.sidebar:
        st.title("🎛️ 增强版控制")
        
        # 模型训练控制
        st.markdown("### 🧠 模型训练")
        
        if st.button("📊 获取一个月历史数据并训练", type="primary"):
            try:
                with st.spinner("📊 正在获取一个月真实历史数据..."):
                    historical_data = api.get_historical_data(days=30)
                    st.session_state.enh_historical_data = historical_data
                    st.success(f"✅ 获取到 {len(historical_data)} 条历史数据")
                
                with st.spinner("🧠 正在训练增强版模型..."):
                    success = model.train_models(historical_data)
                    if success:
                        st.session_state.enh_model_trained = True
                        st.session_state.last_training_time = get_beijing_time().replace(tzinfo=None)
                        st.session_state.auto_refresh_enabled = True
                        st.success("✅ 增强版模型训练成功!")
                        st.success("🔄 自动刷新已启用 (2分钟间隔)")
                    else:
                        st.error("❌ 模型训练失败")
                        
            except Exception as e:
                st.error(f"❌ 数据获取失败: {e}")
        
        # 自动刷新控制
        st.markdown("### 🔄 自动刷新")
        auto_refresh = st.checkbox("启用自动刷新", value=st.session_state.auto_refresh_enabled)
        if auto_refresh != st.session_state.auto_refresh_enabled:
            st.session_state.auto_refresh_enabled = auto_refresh
        
        refresh_interval = "2分钟"  # 默认值
        if auto_refresh:
            refresh_interval = st.selectbox(
                "刷新间隔",
                ["1分钟", "2分钟", "5分钟"],
                index=1
            )
        
        # 定时重训练状态
        st.markdown("### ⏰ 定时重训练")
        if st.session_state.last_training_time:
            time_since_training = get_beijing_time().replace(tzinfo=None) - st.session_state.last_training_time
            minutes_since = int(time_since_training.total_seconds() / 60)
            st.info(f"上次训练: {minutes_since} 分钟前")
            
            # 检查是否需要重训练（每小时）
            if minutes_since >= 60:
                st.warning("⚠️ 建议重新训练模型")
                if st.button("🔄 立即重训练"):
                    try:
                        with st.spinner("🔄 正在重新训练模型..."):
                            success = model.train_models(st.session_state.enh_historical_data)
                            if success:
                                st.session_state.last_training_time = get_beijing_time().replace(tzinfo=None)
                                st.success("✅ 模型重训练完成!")
                    except Exception as e:
                        st.error(f"❌ 重训练失败: {e}")
        
        if st.button("🔄 立即刷新", type="secondary"):
            st.rerun()
    
    # 模型状态显示
    if st.session_state.enh_model_trained:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="success-status">
                <h4>🧠 增强版模型: 已训练</h4>
                <p>📊 训练数据: {model.training_data_count} 条</p>
                <p>📈 特征数量: {len(model.feature_names)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            algorithm_30 = getattr(model, 'best_algorithm_30min', '未知')
            st.markdown(f"""
            <div class="metric-card">
                <h5>📊 30分钟模型</h5>
                <p><strong>最佳算法:</strong> {algorithm_30}</p>
                <p><strong>总体准确率:</strong> {model.accuracy_30min:.1%}</p>
                <p><strong>趋势准确率:</strong> {model.direction_accuracy_30min:.1%}</p>
                <p><strong>F1分数:</strong> {model.mae_30min:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            algorithm_1h = getattr(model, 'best_algorithm_1hour', '未知')
            st.markdown(f"""
            <div class="metric-card">
                <h5>📊 1小时模型</h5>
                <p><strong>最佳算法:</strong> {algorithm_1h}</p>
                <p><strong>总体准确率:</strong> {model.accuracy_1hour:.1%}</p>
                <p><strong>趋势准确率:</strong> {model.direction_accuracy_1hour:.1%}</p>
                <p><strong>F1分数:</strong> {model.mae_1hour:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-status">
            <h3>⚠️ 需要训练模型</h3>
            <p>请点击'获取一个月历史数据并训练'按钮</p>
            <p>系统将获取720条小时级历史数据进行训练</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 获取实时数据并显示
    try:
        with st.spinner("📡 获取实时ETH数据..."):
            price_data = api.get_current_price()
        
        # 价格显示
        st.markdown(f"""
        <div class="main-header">
            <div class="price-display">💎 ${price_data['price']:,.2f}</div>
            <div style="text-align: center; font-size: 1.2rem;">
                📊 24h变化: {price_data['change_24h']:+.2f}%
            </div>
            <div style="text-align: center; font-size: 0.9rem; opacity: 0.8; margin-top: 1rem;">
                🕒 {format_beijing_time()} | 数据源: {', '.join(price_data['apis_used'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 智能预测
        if st.session_state.enh_model_trained and st.session_state.enh_historical_data:
            st.markdown("---")
            st.markdown("### 🧠 增强版AI预测")
            
            try:
                # 准备当前特征
                current_data = st.session_state.enh_historical_data.copy()
                current_data.append({
                    'timestamp': get_beijing_time().replace(tzinfo=None),
                    'price': price_data['price'],
                    'volume': 100000,  # 估算值
                    'high': price_data['price'] * 1.01,
                    'low': price_data['price'] * 0.99,
                    'open': price_data['price']
                })
                
                features, _, _ = model.prepare_features(current_data)
                if len(features) > 0:
                    current_features = features.iloc[-1].values
                    predictions = model.predict(current_features)
                    
                    if predictions:
                        # 🔥 记录历史预测信息
                        current_time = get_beijing_time().replace(tzinfo=None)
                        pred_30min_data = predictions['30min']
                        pred_1hour_data = predictions['1hour']
                        
                        # 计算30分钟预测价格（与目标价格保持一致）
                        if pred_30min_data['class'] > 0:
                            price_change_30 = pred_30min_data['up_prob'] * 0.015
                            target_30min = price_data['price'] * (1 + price_change_30)
                        else:
                            price_change_30 = pred_30min_data['down_prob'] * 0.015
                            target_30min = price_data['price'] * (1 - price_change_30)
                        
                        # 计算1小时预测价格（与目标价格保持一致）
                        if pred_1hour_data['class'] > 0:
                            price_change_1h = pred_1hour_data['up_prob'] * 0.025
                            target_1hour = price_data['price'] * (1 + price_change_1h)
                        else:
                            price_change_1h = pred_1hour_data['down_prob'] * 0.025
                            target_1hour = price_data['price'] * (1 - price_change_1h)
                        
                        # 计算具体预测时间
                        prediction_time_30min = current_time + timedelta(minutes=30)
                        prediction_time_1hour = current_time + timedelta(hours=1)
                        
                        # 计算最高概率
                        max_prob_30 = max(pred_30min_data['up_prob'], pred_30min_data['down_prob']) * 100
                        max_prob_1h = max(pred_1hour_data['up_prob'], pred_1hour_data['down_prob']) * 100
                        
                        # 添加30分钟预测记录
                        prediction_record_30min = {
                            '当前时间': current_time.strftime('%H:%M:%S'),
                            '当前价格': f"${price_data['price']:,.2f}",
                            '预测时间': prediction_time_30min.strftime('%H:%M:%S'),
                            '预测价格': f"${target_30min:,.2f}",
                            '预测涨跌': '📈 上涨' if pred_30min_data['class'] > 0 else '📉 下跌',
                            '预测概率': f"{max_prob_30:.1f}%",
                            '高概率': max_prob_30 > 70
                        }
                        
                        # 添加1小时预测记录
                        prediction_record_1hour = {
                            '当前时间': current_time.strftime('%H:%M:%S'),
                            '当前价格': f"${price_data['price']:,.2f}",
                            '预测时间': prediction_time_1hour.strftime('%H:%M:%S'),
                            '预测价格': f"${target_1hour:,.2f}",
                            '预测涨跌': '📈 上涨' if pred_1hour_data['class'] > 0 else '📉 下跌',
                            '预测概率': f"{max_prob_1h:.1f}%",
                            '高概率': max_prob_1h > 70
                        }
                        
                        # 将记录添加到历史中（最多保留20条）
                        st.session_state.prediction_history.insert(0, prediction_record_30min)
                        st.session_state.prediction_history.insert(1, prediction_record_1hour)
                        if len(st.session_state.prediction_history) > 20:
                            st.session_state.prediction_history = st.session_state.prediction_history[:20]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            pred_30min_data = predictions['30min']
                            class_30 = pred_30min_data['class']
                            up_prob_30 = pred_30min_data['up_prob'] * 100
                            down_prob_30 = pred_30min_data['down_prob'] * 100
                            
                            # 确定趋势和目标价格（二分类）
                            if class_30 > 0:
                                trend_30 = "📈 上涨"
                                trend_color_30 = "#00C851"
                                price_change_est = up_prob_30 / 100 * 0.015  # 基于概率估算1.5%变化
                                target_30min = price_data['price'] * (1 + price_change_est)
                            else:
                                trend_30 = "📉 下跌"
                                trend_color_30 = "#FF4444"
                                price_change_est = down_prob_30 / 100 * 0.015
                                target_30min = price_data['price'] * (1 - price_change_est)
                            
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 4px solid {trend_color_30};">
                                <h4>📊 30分钟预测</h4>
                                <p><strong>趋势预测:</strong> {trend_30} (类别: {class_30})</p>
                                <p><strong>上涨概率:</strong> {up_prob_30:.1f}%</p>
                                <p><strong>下跌概率:</strong> {down_prob_30:.1f}%</p>
                                <p><strong>目标价格:</strong> ${target_30min:,.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            pred_1hour_data = predictions['1hour']
                            class_1h = pred_1hour_data['class']
                            up_prob_1h = pred_1hour_data['up_prob'] * 100
                            down_prob_1h = pred_1hour_data['down_prob'] * 100
                            
                            # 确定趋势和目标价格（二分类）
                            if class_1h > 0:
                                trend_1h = "📈 上涨"
                                trend_color_1h = "#00C851"
                                price_change_est = up_prob_1h / 100 * 0.025  # 基于概率估算2.5%变化
                                target_1hour = price_data['price'] * (1 + price_change_est)
                            else:
                                trend_1h = "📉 下跌"
                                trend_color_1h = "#FF4444"
                                price_change_est = down_prob_1h / 100 * 0.025
                                target_1hour = price_data['price'] * (1 - price_change_est)
                            
                            st.markdown(f"""
                            <div class="metric-card" style="border-left: 4px solid {trend_color_1h};">
                                <h4>📊 1小时预测</h4>
                                <p><strong>趋势预测:</strong> {trend_1h} (类别: {class_1h})</p>
                                <p><strong>上涨概率:</strong> {up_prob_1h:.1f}%</p>
                                <p><strong>下跌概率:</strong> {down_prob_1h:.1f}%</p>
                                <p><strong>目标价格:</strong> ${target_1hour:,.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 交易信号 - 基于二分类预测
                        max_prob_30 = max(up_prob_30, down_prob_30)
                        max_prob_1h = max(up_prob_1h, down_prob_1h)
                        
                        # 只有在概率足够高时才显示交易信号
                        if max_prob_30 > 60:
                            if up_prob_30 > down_prob_30:
                                signal_action = "💰 建议买涨"
                                signal_reason = f"上涨概率 {up_prob_30:.1f}%"
                                signal_color = "#00C851"
                                target_signal = target_30min
                            else:
                                signal_action = "💸 建议买跌"
                                signal_reason = f"下跌概率 {down_prob_30:.1f}%"
                                signal_color = "#FF4444"
                                target_signal = target_30min
                            
                            confidence_level = "高" if max_prob_30 > 75 else "中"
                            
                            st.markdown(f"""
                            <div class="success-status" style="border-left: 4px solid {signal_color};">
                                <h4>🚨 30分钟交易信号</h4>
                                <p><strong>{signal_action}</strong> (置信度: {confidence_level})</p>
                                <p>{signal_reason}，目标价格 ${target_signal:,.2f}</p>
                                <p>信号强度: {'🔥' * min(5, int(max_prob_30/20))}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 🤖 AI智能决策信号
                        st.markdown("---")
                        st.markdown("### 🤖 AI智能决策系统")
                        
                        ai_signal = model.generate_trading_signal(predictions, price_data['price'])
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="training-card" style="background: linear-gradient(135deg, {ai_signal['color']}, {ai_signal['color']}CC); margin: 1rem 0;">
                                <h3 style="text-align: center; margin-bottom: 1rem;">
                                    {ai_signal['action']}
                                </h3>
                                <div style="background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                                    <p><strong>🎯 {ai_signal['priority']}</strong></p>
                                    <p><strong>📊 置信度:</strong> {ai_signal['confidence']}</p>
                                    <p><strong>💡 分析理由:</strong> {ai_signal['reason']}</p>
                                    <p><strong>🎪 目标价格:</strong> ${ai_signal['target_price']:,.2f}</p>
                                    <p><strong>📈 信号强度:</strong> {ai_signal['signal_strength']}/100</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### 📋 决策详情")
                            details = ai_signal['details']
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <p><strong>30分钟预测:</strong> {details['30min_prediction']}</p>
                                <p><strong>1小时预测:</strong> {details['1hour_prediction']}</p>
                                <p><strong>方向一致性:</strong> {'✅ 一致' if details['direction_consistent'] else '❌ 分歧'}</p>
                                <p><strong>30分钟准确率:</strong> {details['model_accuracy_30min']}</p>
                                <p><strong>1小时准确率:</strong> {details['model_accuracy_1hour']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 风险提示
                        if ai_signal['confidence'] in ['高', '中']:
                            st.info(f"💡 AI建议: {ai_signal['action']} | 请结合其他技术分析指标和市场情况综合判断")
                        else:
                            st.warning("⚠️ 当前市场信号不明确，建议继续观望等待更好的入场时机")
                        
                        # 🔥 合约交易点位分析
                        st.markdown("---")
                        st.markdown("### 💼 专业合约交易点位分析")
                        
                        def calculate_trading_levels(current_price, predictions, historical_data):
                            """计算专业交易点位"""
                            # 基础参数
                            pred_30min = predictions['30min']
                            pred_1hour = predictions['1hour']
                            
                            # 计算平均波动率（基于历史数据）
                            df_hist = pd.DataFrame(historical_data)
                            price_changes = df_hist['price'].pct_change().dropna()
                            daily_volatility = price_changes.std() * np.sqrt(24)  # 日波动率
                            
                            # 计算技术支撑阻力位
                            recent_prices = df_hist['price'].tail(48).values  # 近48小时
                            support_level = np.percentile(recent_prices, 25)  # 25%分位数作为支撑
                            resistance_level = np.percentile(recent_prices, 75)  # 75%分位数作为阻力
                            
                            # 综合AI预测强度
                            avg_up_prob = (pred_30min['up_prob'] + pred_1hour['up_prob']) / 2
                            avg_down_prob = (pred_30min['down_prob'] + pred_1hour['down_prob']) / 2
                            signal_strength = abs(avg_up_prob - avg_down_prob)
                            
                            # 确定主要方向
                            is_bullish = avg_up_prob > avg_down_prob
                            confidence = max(avg_up_prob, avg_down_prob)
                            
                            if is_bullish:
                                # 做多策略
                                entry_price = current_price * (1 - 0.002)  # 略低入场，等待回调
                                
                                # 止盈目标（基于预测强度和波动率）
                                if confidence > 0.75:
                                    take_profit_1 = current_price * (1 + 0.015)  # 1.5%
                                    take_profit_2 = current_price * (1 + 0.025)  # 2.5%
                                    take_profit_3 = current_price * (1 + 0.04)   # 4%
                                elif confidence > 0.65:
                                    take_profit_1 = current_price * (1 + 0.01)
                                    take_profit_2 = current_price * (1 + 0.018)
                                    take_profit_3 = current_price * (1 + 0.028)
                                else:
                                    take_profit_1 = current_price * (1 + 0.008)
                                    take_profit_2 = current_price * (1 + 0.015)
                                    take_profit_3 = current_price * (1 + 0.022)
                                
                                # 止损点位
                                stop_loss = max(
                                    current_price * (1 - 0.012),  # 1.2%止损
                                    support_level * 0.998  # 支撑位下方
                                )
                                
                                action = "🚀 建议做多"
                                action_color = "#00C851"
                                
                            else:
                                # 做空策略
                                entry_price = current_price * (1 + 0.002)  # 略高入场，等待反弹
                                
                                # 止盈目标
                                if confidence > 0.75:
                                    take_profit_1 = current_price * (1 - 0.015)
                                    take_profit_2 = current_price * (1 - 0.025)
                                    take_profit_3 = current_price * (1 - 0.04)
                                elif confidence > 0.65:
                                    take_profit_1 = current_price * (1 - 0.01)
                                    take_profit_2 = current_price * (1 - 0.018)
                                    take_profit_3 = current_price * (1 - 0.028)
                                else:
                                    take_profit_1 = current_price * (1 - 0.008)
                                    take_profit_2 = current_price * (1 - 0.015)
                                    take_profit_3 = current_price * (1 - 0.022)
                                
                                # 止损点位
                                stop_loss = min(
                                    current_price * (1 + 0.012),  # 1.2%止损
                                    resistance_level * 1.002  # 阻力位上方
                                )
                                
                                action = "📉 建议做空"
                                action_color = "#FF4444"
                            
                            return {
                                'action': action,
                                'action_color': action_color,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit_1': take_profit_1,
                                'take_profit_2': take_profit_2,
                                'take_profit_3': take_profit_3,
                                'confidence': confidence,
                                'signal_strength': signal_strength,
                                'support_level': support_level,
                                'resistance_level': resistance_level,
                                'daily_volatility': daily_volatility,
                                'risk_reward_1': abs(take_profit_1 - current_price) / abs(stop_loss - current_price),
                                'risk_reward_2': abs(take_profit_2 - current_price) / abs(stop_loss - current_price),
                                'risk_reward_3': abs(take_profit_3 - current_price) / abs(stop_loss - current_price)
                            }
                        
                        # 计算交易点位
                        if st.session_state.enh_historical_data:
                            trading_analysis = calculate_trading_levels(
                                price_data['price'], 
                                predictions, 
                                st.session_state.enh_historical_data
                            )
                            
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                # 主要交易建议
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {trading_analysis['action_color']}, {trading_analysis['action_color']}CC); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
                                    <h3>{trading_analysis['action']}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 关键交易点位
                                st.markdown("#### 📊 关键交易点位")
                                
                                col1_1, col1_2 = st.columns(2)
                                with col1_1:
                                    st.metric("🎯 建议入场", f"${trading_analysis['entry_price']:,.2f}")
                                    st.metric("🛡️ 止损点位", f"${trading_analysis['stop_loss']:,.2f}")
                                
                                with col1_2:
                                    st.metric("💰 止盈目标1", f"${trading_analysis['take_profit_1']:,.2f}")
                                    st.metric("💎 止盈目标2", f"${trading_analysis['take_profit_2']:,.2f}")
                                
                                st.metric("🚀 止盈目标3", f"${trading_analysis['take_profit_3']:,.2f}")
                                
                                # 风险收益比
                                st.markdown("#### 📈 风险收益比")
                                col1_1, col1_2, col1_3 = st.columns(3)
                                with col1_1:
                                    st.metric("目标1", f"{trading_analysis['risk_reward_1']:.2f}:1")
                                with col1_2:
                                    st.metric("目标2", f"{trading_analysis['risk_reward_2']:.2f}:1")
                                with col1_3:
                                    st.metric("目标3", f"{trading_analysis['risk_reward_3']:.2f}:1")
                            
                            with col2:
                                st.markdown("#### 📊 技术分析要素")
                                
                                st.metric("📊 AI置信度", f"{trading_analysis['confidence']*100:.1f}%")
                                st.metric("⚡ 信号强度", f"{trading_analysis['signal_strength']*100:.1f}%")
                                st.metric("📈 日波动率", f"{trading_analysis['daily_volatility']*100:.2f}%")
                                st.metric("🔻 技术支撑", f"${trading_analysis['support_level']:,.2f}")
                                st.metric("🔺 技术阻力", f"${trading_analysis['resistance_level']:,.2f}")
                                
                                # 交易建议等级
                                if trading_analysis['confidence'] > 0.75 and trading_analysis['signal_strength'] > 0.3:
                                    grade = "🔥 强烈推荐"
                                    grade_color = "#FF6B35"
                                elif trading_analysis['confidence'] > 0.65 and trading_analysis['signal_strength'] > 0.2:
                                    grade = "⚡ 推荐"
                                    grade_color = "#4ECDC4"
                                elif trading_analysis['confidence'] > 0.55:
                                    grade = "⚠️ 谨慎考虑"
                                    grade_color = "#FFD93D"
                                else:
                                    grade = "❌ 不建议"
                                    grade_color = "#FF6B6B"
                                
                                st.markdown(f"""
                                <div style="background: {grade_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                                    <h4>交易推荐等级</h4>
                                    <p style="font-size: 1.3rem; font-weight: bold; margin: 0;">{grade}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # 风险提示
                            st.warning("""
                            ⚠️ **重要风险提示**
                            - 以上点位仅供参考，请结合市场实际情况调整
                            - 建议分批进场，控制仓位大小  
                            - 严格执行止损，保护本金安全
                            - 市场有风险，投资需谨慎
                            """)
            
            except Exception as e:
                st.error(f"预测过程出错: {e}")
        
        # 历史数据图表
        if st.session_state.enh_historical_data:
            st.markdown("---")
            
            # 🔥 历史预测信息日志
            if st.session_state.prediction_history:
                st.markdown("### 📋 历史预测信息日志")
                
                # 检查是否有高概率预测
                high_prob_count = sum(1 for record in st.session_state.prediction_history if record.get('高概率', False))
                if high_prob_count > 0:
                    st.success(f"🚨 特别提醒：发现 {high_prob_count} 条高概率预测（>70%）！请重点关注！")
                
                # 创建预测历史表格
                df_predictions = pd.DataFrame(st.session_state.prediction_history)
                
                # 移除内部使用的'高概率'字段，不在表格中显示
                display_df = df_predictions.drop(columns=['高概率'] if '高概率' in df_predictions.columns else [])
                
                # 直接显示表格，不使用复杂样式
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "当前时间": st.column_config.TextColumn("🕒 当前时间", width="small"),
                        "当前价格": st.column_config.TextColumn("💰 当前价格", width="medium"),
                        "预测时间": st.column_config.TextColumn("⏰ 预测时间", width="small"),
                        "预测价格": st.column_config.TextColumn("🎯 预测价格", width="medium"),
                        "预测涨跌": st.column_config.TextColumn("📈 预测涨跌", width="small"),
                        "预测概率": st.column_config.TextColumn("📊 预测概率", width="small")
                    }
                )
                
                # 统计信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 预测记录数", len(st.session_state.prediction_history))
                with col2:
                    up_predictions = sum(1 for record in st.session_state.prediction_history if '📈' in record['预测涨跌'])
                    st.metric("📈 看涨次数", up_predictions)
                with col3:
                    down_predictions = len(st.session_state.prediction_history) - up_predictions
                    st.metric("📉 看跌次数", down_predictions)
                with col4:
                    avg_prob = sum(float(record['预测概率'].replace('%', '')) for record in st.session_state.prediction_history) / len(st.session_state.prediction_history)
                    st.metric("📊 平均概率", f"{avg_prob:.1f}%")
            
            st.markdown("### 📈 一个月历史价格走势 (K线图)")
            
            df_hist = pd.DataFrame(st.session_state.enh_historical_data)
            
            # 创建子图：K线图 + 成交量图
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('ETH/USDT K线图', '成交量'),
                row_heights=[0.7, 0.3]
            )
            
            # 添加K线图（蜡烛图）
            fig.add_trace(go.Candlestick(
                x=df_hist['timestamp'],
                open=df_hist['open'],
                high=df_hist['high'],
                low=df_hist['low'],
                close=df_hist['price'],
                name='ETH/USDT',
                increasing_line_color='#00C851',  # 上涨蜡烛颜色
                decreasing_line_color='#FF4444',  # 下跌蜡烛颜色
                increasing_fillcolor='#00C851',
                decreasing_fillcolor='#FF4444'
            ), row=1, col=1)
            
            # 添加移动平均线
            if len(df_hist) >= 20:
                ma20 = df_hist['price'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=df_hist['timestamp'],
                    y=ma20,
                    mode='lines',
                    name='MA20',
                    line=dict(color='#FFB300', width=1.5),
                    opacity=0.8
                ), row=1, col=1)
            
            if len(df_hist) >= 50:
                ma50 = df_hist['price'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=df_hist['timestamp'],
                    y=ma50,
                    mode='lines',
                    name='MA50',
                    line=dict(color='#9C27B0', width=1.5),
                    opacity=0.8
                ), row=1, col=1)
            
            # 添加成交量柱状图
            colors = ['#00C851' if close >= open_price else '#FF4444' 
                     for close, open_price in zip(df_hist['price'], df_hist['open'])]
            
            fig.add_trace(go.Bar(
                x=df_hist['timestamp'],
                y=df_hist['volume'],
                name='成交量',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
            
            # 更新布局
            fig.update_layout(
                title="ETH 一个月历史价格走势 (720条小时级K线数据)",
                template='plotly_white',
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False  # 隐藏K线图下方的范围滑块
            )
            
            # 更新X轴和Y轴
            fig.update_xaxes(title_text="时间", row=2, col=1)
            fig.update_yaxes(title_text="价格 (USD)", row=1, col=1)
            fig.update_yaxes(title_text="成交量", row=2, col=1)
            
            # 移除非交易时间的空白（可选）
            fig.update_xaxes(
                rangebreaks=[
                    # 注释：这里可以添加市场休市时间的空白，但加密货币24/7交易，所以不需要
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ 数据获取失败: {e}")
    
    # 系统状态
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🕒 当前时间", format_beijing_time())
    with col2:
        runtime = get_beijing_time().replace(tzinfo=None) - st.session_state.system_start_time
        st.metric("⏱️ 运行时间", f"{int(runtime.total_seconds()/60)}分钟")
    with col3:
        model_status = "已训练" if st.session_state.enh_model_trained else "未训练"
        st.metric("🧠 模型状态", model_status)
    with col4:
        refresh_status = "开启" if st.session_state.auto_refresh_enabled else "关闭"
        st.metric("🔄 自动刷新", refresh_status)
    
    # 自动刷新逻辑 - 使用专业组件
    if st.session_state.auto_refresh_enabled and st.session_state.enh_model_trained and AUTOREFRESH_AVAILABLE:
        # 获取刷新间隔（毫秒）
        interval_map = {"1分钟": 60000, "2分钟": 120000, "5分钟": 300000}
        interval_ms = interval_map.get(refresh_interval, 120000)
        
        # 检查是否需要重训练
        if st.session_state.last_training_time:
            time_since_training = get_beijing_time().replace(tzinfo=None) - st.session_state.last_training_time
            if time_since_training.total_seconds() >= 3600:  # 1小时
                try:
                    with st.spinner("🔄 正在自动重训练模型..."):
                        model.train_models(st.session_state.enh_historical_data)
                        st.session_state.last_training_time = get_beijing_time().replace(tzinfo=None)
                        st.success("✅ 模型已自动重训练完成!")
                except Exception as e:
                    st.error(f"❌ 自动重训练失败: {e}")
        
        # 显示自动刷新状态
        st.success(f"🔄 自动刷新已启用，每{refresh_interval}自动更新")
        
        # 使用专业的自动刷新组件
        count = st_autorefresh(interval=interval_ms, key="eth_monitor_refresh")
        
        # 显示刷新次数和运行状态
        if count == 0:
            st.info("🚀 自动刷新系统已启动")
        else:
            st.info(f"🔄 已自动刷新 {count} 次 | 系统运行正常")
    
    elif st.session_state.auto_refresh_enabled and not AUTOREFRESH_AVAILABLE:
        # 如果自动刷新组件不可用，显示手动刷新提示
        st.warning("⚠️ 自动刷新组件未安装，请手动点击侧边栏的'🔄 立即刷新'按钮")
        st.info("💡 要启用自动刷新，请运行: pip install streamlit-autorefresh")

if __name__ == "__main__":
    main() 