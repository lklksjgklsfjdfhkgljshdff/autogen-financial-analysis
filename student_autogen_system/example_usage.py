"""
学生级AutoGen金融分析系统使用示例
"""

from main_system import StudentAutoGenSystem
import json

def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")

    # 创建分析系统
    # 注意：需要替换为您的真实API密钥
    api_key = "your-openai-api-key"

    # 如果没有API密钥，可以使用简化模式
    if api_key == "your-openai-api-key":
        print("⚠️  请先配置您的OpenAI API密钥")
        print("提示：访问 https://openai.com/ 获取API密钥")
        return

    system = StudentAutoGenSystem(api_key)

    # 分析苹果公司股票
    symbol = "AAPL"
    print(f"正在分析 {symbol}...")

    result = system.analyze_stock(symbol)

    if result["status"] == "success":
        print("✅ 分析成功！")
        print("\n📊 分析报告：")
        print(result["report"])

        print("\n🤖 AI分析结果：")
        print(result["ai_analysis"])

        # 保存结果到文件
        with open(f"{symbol}_analysis.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 分析结果已保存到 {symbol}_analysis.json")

    else:
        print(f"❌ 分析失败：{result['message']}")

def example_multiple_stocks():
    """多只股票分析示例"""
    print("\n=== 多只股票分析示例 ===")

    # 示例股票列表
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

    api_key = "your-openai-api-key"
    if api_key == "your-openai-api-key":
        print("⚠️  需要配置API密钥才能运行此示例")
        return

    system = StudentAutoGenSystem(api_key)

    results = {}
    for symbol in symbols:
        print(f"\n正在分析 {symbol}...")
        result = system.analyze_stock(symbol)
        results[symbol] = result

        if result["status"] == "success":
            print(f"✅ {symbol} 分析成功")
            print(f"   ROE: {result['financial_metrics']['roe']:.2f}%")
            print(f"   资产负债率: {result['financial_metrics']['debt_ratio']:.2f}%")
            print(f"   波动率: {result['risk_metrics']['volatility']:.2f}%")
        else:
            print(f"❌ {symbol} 分析失败：{result['message']}")

    # 保存所有结果
    with open("multi_stock_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n💾 多股票分析结果已保存到 multi_stock_analysis.json")

def example_financial_metrics_only():
    """仅计算财务指标示例（无需API密钥）"""
    print("\n=== 财务指标计算示例 ===")

    from main_system import SimpleDataCollector, SimpleFinancialAnalyzer

    # 创建数据收集器和分析器
    data_collector = SimpleDataCollector()
    financial_analyzer = SimpleFinancialAnalyzer()

    # 收集数据
    symbol = "AAPL"
    print(f"正在收集 {symbol} 的数据...")
    data = data_collector.collect_stock_data(symbol)

    if data:
        print("✅ 数据收集成功")

        # 计算财务指标
        metrics = financial_analyzer.calculate_basic_metrics(data)

        print("\n📈 财务指标：")
        print(f"净资产收益率 (ROE): {metrics.roe:.2f}%")
        print(f"总资产收益率 (ROA): {metrics.roa:.2f}%")
        print(f"资产负债率: {metrics.debt_ratio:.2f}%")
        print(f"毛利率: {metrics.gross_margin:.2f}%")
        print(f"净利率: {metrics.net_margin:.2f}%")
        print(f"流动比率: {metrics.current_ratio:.2f}")

        # 简单评价
        print("\n📝 简单评价：")
        if metrics.roe > 15:
            print("• 公司盈利能力优秀")
        elif metrics.roe > 10:
            print("• 公司盈利能力良好")
        else:
            print("• 公司盈利能力有待提升")

        if metrics.debt_ratio < 30:
            print("• 财务杠杆较低，偿债能力强")
        elif metrics.debt_ratio < 60:
            print("• 财务杠杆适中")
        else:
            print("• 财务杠杆较高，需关注偿债风险")

    else:
        print("❌ 数据收集失败")

def example_risk_analysis():
    """风险分析示例（无需API密钥）"""
    print("\n=== 风险分析示例 ===")

    from main_system import SimpleDataCollector, SimpleRiskAnalyzer

    # 创建数据收集器和风险分析器
    data_collector = SimpleDataCollector()
    risk_analyzer = SimpleRiskAnalyzer()

    # 收集数据
    symbol = "AAPL"
    print(f"正在收集 {symbol} 的数据...")
    data = data_collector.collect_stock_data(symbol)

    if data:
        print("✅ 数据收集成功")

        # 计算风险指标
        risk_metrics = risk_analyzer.calculate_basic_risk(data)

        print("\n⚠️  风险指标：")
        print(f"年化波动率: {risk_metrics.volatility:.2f}%")
        print(f"95% VaR: {risk_metrics.var_95:.2f}%")
        print(f"最大回撤: {risk_metrics.max_drawdown:.2f}%")

        # 风险评价
        print("\n📊 风险评价：")
        if risk_metrics.volatility < 20:
            print("• 价格波动相对较小，风险较低")
        elif risk_metrics.volatility < 35:
            print("• 价格波动适中，风险中等")
        else:
            print("• 价格波动较大，风险较高")

        if risk_metrics.max_drawdown > -0.3:
            print("• 历史最大回撤相对较小")
        else:
            print("• 历史最大回撤较大，需注意风险")

    else:
        print("❌ 数据收集失败")

def main():
    """主函数 - 运行所有示例"""
    print("🎓 学生级AutoGen金融分析系统 - 使用示例")
    print("=" * 50)

    # 运行示例
    example_financial_metrics_only()
    example_risk_analysis()

    # 需要API密钥的示例
    print("\n" + "=" * 50)
    print("注意：以下示例需要OpenAI API密钥")
    example_basic_usage()
    example_multiple_stocks()

    print("\n" + "=" * 50)
    print("🎉 所有示例运行完成！")
    print("\n💡 提示：")
    print("1. 请确保已安装所有依赖：pip install -r requirements.txt")
    print("2. 配置您的OpenAI API密钥以体验完整功能")
    print("3. 查看README.md了解更多使用方法")

if __name__ == "__main__":
    main()