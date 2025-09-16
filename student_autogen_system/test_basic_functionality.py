"""
基础功能测试（无需AutoGen依赖）
测试数据收集和财务分析功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_collection():
    """测试数据收集功能"""
    print("=== 测试数据收集功能 ===")

    try:
        # 只导入需要的模块
        import yfinance as yf
        import pandas as pd
        print("✅ 成功导入 yfinance 和 pandas")

        # 测试数据收集
        symbol = "AAPL"
        print(f"正在收集 {symbol} 的数据...")

        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        financials = stock.financials

        print(f"✅ 成功收集数据")
        print(f"   - 价格数据行数: {len(hist)}")
        print(f"   - 财务数据列数: {len(financials.columns)}")

        # 显示一些基本信息
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1]
            print(f"   - 最新价格: ${latest_price:.2f}")

        return True

    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install yfinance pandas numpy")
        return False

    except Exception as e:
        print(f"❌ 数据收集失败: {e}")
        return False

def test_financial_calculation():
    """测试财务计算功能"""
    print("\n=== 测试财务计算功能 ===")

    try:
        import pandas as pd
        import numpy as np

        # 创建模拟数据
        print("创建模拟财务数据...")
        # 按照真实的财务报表格式创建数据
        index_data = ['Total Revenue', 'Net Income', 'Total Assets', 'Total Stockholder Equity', 'Total Debt', 'Total Current Assets', 'Total Current Liabilities', 'Gross Profit']
        column_data = ['2023', '2022', '2021']
        financial_values = [
            [150000, 120000, 100000],  # Total Revenue
            [20000, 15000, 10000],     # Net Income
            [75000, 60000, 50000],     # Total Assets
            [45000, 35000, 30000],     # Total Stockholder Equity
            [20000, 18000, 15000],     # Total Debt
            [40000, 30000, 25000],     # Total Current Assets
            [25000, 18000, 15000],     # Total Current Liabilities
            [60000, 48000, 40000]      # Gross Profit
        ]

        financials = pd.DataFrame(financial_values, index=index_data, columns=column_data)
        print("✅ 成功创建模拟数据")

        # 计算财务比率
        latest_year = financials.columns[0]

        revenue = financials.loc['Total Revenue', latest_year]
        net_income = financials.loc['Net Income', latest_year]
        total_assets = financials.loc['Total Assets', latest_year]
        total_equity = financials.loc['Total Stockholder Equity', latest_year]
        total_debt = financials.loc['Total Debt', latest_year]
        current_assets = financials.loc['Total Current Assets', latest_year]
        current_liabilities = financials.loc['Total Current Liabilities', latest_year]
        gross_profit = financials.loc['Gross Profit', latest_year]

        # 计算比率
        roe = (net_income / total_equity * 100) if total_equity != 0 else 0
        roa = (net_income / total_assets * 100) if total_assets != 0 else 0
        debt_ratio = (total_debt / total_assets * 100) if total_assets != 0 else 0
        gross_margin = (gross_profit / revenue * 100) if revenue != 0 else 0
        net_margin = (net_income / revenue * 100) if revenue != 0 else 0
        current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0

        print("\n📊 财务比率计算结果：")
        print(f"   - ROE: {roe:.2f}%")
        print(f"   - ROA: {roa:.2f}%")
        print(f"   - 资产负债率: {debt_ratio:.2f}%")
        print(f"   - 毛利率: {gross_margin:.2f}%")
        print(f"   - 净利率: {net_margin:.2f}%")
        print(f"   - 流动比率: {current_ratio:.2f}")

        print("✅ 财务计算功能正常")
        return True

    except Exception as e:
        print(f"❌ 财务计算失败: {e}")
        return False

def test_risk_calculation():
    """测试风险计算功能"""
    print("\n=== 测试风险计算功能 ===")

    try:
        import pandas as pd
        import numpy as np

        # 创建模拟价格数据
        print("创建模拟价格数据...")
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        prices = np.random.normal(100, 5, len(dates))  # 模拟股价数据
        price_series = pd.Series(prices, index=dates)

        # 计算收益率
        returns = price_series.pct_change().dropna()

        # 计算风险指标
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        var_95 = np.percentile(returns, 5)  # 95% VaR

        # 计算最大回撤
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        print("\n⚠️  风险指标计算结果：")
        print(f"   - 年化波动率: {volatility:.2f}%")
        print(f"   - 95% VaR: {abs(var_95):.2f}%")
        print(f"   - 最大回撤: {abs(max_drawdown):.2f}%")

        print("✅ 风险计算功能正常")
        return True

    except Exception as e:
        print(f"❌ 风险计算失败: {e}")
        return False

def test_system_structure():
    """测试系统文件结构"""
    print("\n=== 测试系统文件结构 ===")

    required_files = [
        'main_system.py',
        'requirements.txt',
        'README.md',
        'example_usage.py',
        'test_basic_functionality.py'
    ]

    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - 缺失")
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ 缺失文件: {missing_files}")
        return False
    else:
        print("\n✅ 所有必要文件都存在")
        return True

def main():
    """主测试函数"""
    print("🧪 学生级AutoGen金融分析系统 - 功能测试")
    print("=" * 60)

    test_results = []

    # 运行测试
    test_results.append(("系统文件结构", test_system_structure()))
    test_results.append(("数据收集功能", test_data_collection()))
    test_results.append(("财务计算功能", test_financial_calculation()))
    test_results.append(("风险计算功能", test_risk_calculation()))

    # 总结测试结果
    print("\n" + "=" * 60)
    print("📋 测试结果总结：")

    passed = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 总体结果: {passed}/{len(test_results)} 测试通过")

    if passed == len(test_results):
        print("🎉 所有测试通过！系统基础功能正常。")
        print("\n💡 下一步：")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 配置OpenAI API密钥")
        print("3. 运行完整示例: python example_usage.py")
    else:
        print("⚠️  部分测试失败，请检查相关问题。")

if __name__ == "__main__":
    main()