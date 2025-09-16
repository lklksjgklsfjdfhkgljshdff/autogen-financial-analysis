"""
Main Application Entry Point
AutoGen Financial Analysis System
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.agent_factory import FinancialAgentFactory
from agents.agent_orchestrator import AgentOrchestrator
from data.data_collector import EnterpriseDataCollector
from analysis.financial_analyzer import AdvancedFinancialAnalyzer
from risk.risk_analyzer import AdvancedRiskAnalyzer
from quant.portfolio_optimizer import QuantitativeAnalyzer
from reports.report_generator import ReportGenerator, get_report_generator
from reports.visualizer import Visualizer, get_visualizer
from reports.data_formatter import DataFormatter, get_data_formatter
from reports.export_manager import ExportManager, ExportConfig, ExportFormat, get_export_manager
from config.config_manager import ConfigurationManager
from monitoring.monitoring_system import SystemMonitor
from monitoring.logging_system import LoggingSystem
from security.security_manager import SecurityManager
from security.error_handler import ErrorHandler
from performance.performance_manager import PerformanceManager
from cache.cache_manager import CacheManager


class AutoGenFinancialAnalysisSystem:
    """AutoGen金融分析系统主类"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_manager = ConfigurationManager(config_path)
        self.logger = None
        self.system_monitor = None
        self.security_manager = None
        self.error_handler = None
        self.performance_manager = None
        self.cache_manager = None
        self.data_collector = None
        self.financial_analyzer = None
        self.risk_analyzer = None
        self.quant_analyzer = None
        self.report_generator = None
        self.visualizer = None
        self.data_formatter = None
        self.export_manager = None
        self.agent_factory = None
        self.agent_orchestrator = None

    async def initialize(self) -> None:
        """初始化系统"""
        try:
            # 初始化日志系统
            log_level = self.config_manager.get('logging.level', 'INFO')
            log_file = self.config_manager.get('logging.file', 'autogen_analysis.log')

            logging_system = LoggingSystem()
            logging_system.setup_logging(
                level=getattr(logging, log_level),
                log_file=log_file
            )
            self.logger = logging.getLogger(__name__)

            self.logger.info("正在初始化AutoGen金融分析系统...")

            # 初始化监控系统
            self.system_monitor = SystemMonitor()
            self.system_monitor.start_monitoring()

            # 初始化安全管理器
            security_config = self.config_manager.get('security', {})
            self.security_manager = SecurityManager(security_config.get('secret_key', 'default-secret'))

            # 初始化错误处理器
            self.error_handler = ErrorHandler()
            self._setup_error_handlers()

            # 初始化性能管理器
            perf_config = self.config_manager.get('performance', {})
            self.performance_manager = PerformanceManager(perf_config)

            # 初始化缓存管理器
            cache_config = self.config_manager.get('cache', {})
            redis_url = cache_config.get('redis_url', 'redis://localhost:6379')
            self.cache_manager = CacheManager(redis_url)

            # 初始化数据收集器
            api_keys = self.config_manager.get('api_keys', {})
            self.data_collector = EnterpriseDataCollector(api_keys, redis_url)

            # 初始化分析器
            self.financial_analyzer = AdvancedFinancialAnalyzer()
            self.risk_analyzer = AdvancedRiskAnalyzer()
            self.quant_analyzer = QuantitativeAnalyzer()

            # 初始化报告生成器
            self.report_generator = get_report_generator()
            self.visualizer = get_visualizer()
            self.data_formatter = get_data_formatter()
            self.export_manager = get_export_manager()

            # 初始化智能体系统
            llm_config = self.config_manager.get('llm', {})
            self.agent_factory = FinancialAgentFactory(llm_config)
            self.agent_orchestrator = AgentOrchestrator({})

            self.logger.info("系统初始化完成")

        except Exception as e:
            if self.logger:
                self.logger.error(f"系统初始化失败: {str(e)}")
            else:
                print(f"系统初始化失败: {str(e)}")
            raise

    def _setup_error_handlers(self) -> None:
        """设置错误处理器"""
        # 注册重试策略
        self.error_handler.register_retry_strategy(ConnectionError, max_retries=3)
        self.error_handler.register_retry_strategy(TimeoutError, max_retries=2)

        # 注册错误回调
        def on_data_error(error, *args, **kwargs):
            self.logger.error(f"数据收集错误: {str(error)}")

        def on_analysis_error(error, *args, **kwargs):
            self.logger.error(f"分析错误: {str(error)}")

        self.error_handler.register_error_callback(ValueError, on_data_error)
        self.error_handler.register_error_callback(RuntimeError, on_analysis_error)

    async def analyze_company(self, symbol: str,
                            analysis_type: str = "comprehensive",
                            export_formats: List[str] = None,
                            output_dir: str = "output") -> Dict[str, Any]:
        """分析公司"""
        try:
            self.logger.info(f"开始分析 {symbol}")

            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 收集数据
            self.logger.info("收集数据中...")
            data = await self.data_collector.collect_comprehensive_data(symbol)

            # 执行分析
            self.logger.info("执行财务分析...")
            financial_metrics = self.financial_analyzer.calculate_comprehensive_metrics(data)

            self.logger.info("执行风险分析...")
            market_data = data.get('market_data', {}).get('price_history')
            if market_data is not None:
                risk_metrics = self.risk_analyzer.calculate_comprehensive_risk_metrics(market_data)
            else:
                self.logger.warning("无法获取市场数据，跳过风险分析")
                risk_metrics = {}

            self.logger.info("执行量化分析...")
            # 这里可以添加量化分析逻辑
            quant_results = {}

            # 生成报告
            self.logger.info("生成分析报告...")
            report_data = {
                'symbol': symbol,
                'analysis_date': datetime.now().isoformat(),
                'analysis_type': analysis_type,
                'financial_metrics': financial_metrics.to_dict() if hasattr(financial_metrics, 'to_dict') else financial_metrics,
                'risk_metrics': risk_metrics.to_dict() if hasattr(risk_metrics, 'to_dict') else risk_metrics,
                'quant_results': quant_results,
                'data_quality': data.get('data_quality', {}),
                'summary': self._generate_summary(financial_metrics, risk_metrics),
                'recommendations': self._generate_recommendations(financial_metrics, risk_metrics)
            }

            # 导出报告
            if export_formats:
                await self._export_reports(report_data, export_formats, output_path, symbol)

            self.logger.info(f"分析完成: {symbol}")
            return report_data

        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            raise

    async def analyze_portfolio(self, symbols: List[str],
                              portfolio_weights: Dict[str, float] = None,
                              export_formats: List[str] = None,
                              output_dir: str = "output") -> Dict[str, Any]:
        """分析投资组合"""
        try:
            self.logger.info(f"开始分析投资组合: {symbols}")

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 并行分析多个公司
            tasks = []
            for symbol in symbols:
                task = self.analyze_company(symbol, export_formats=None)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            portfolio_data = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    self.logger.error(f"{symbol} 分析失败: {str(result)}")
                    continue
                portfolio_data[symbol] = result

            # 投资组合分析
            self.logger.info("执行投资组合分析...")
            portfolio_analysis = await self._analyze_portfolio_metrics(portfolio_data, portfolio_weights)

            # 生成投资组合报告
            portfolio_report = {
                'portfolio_symbols': symbols,
                'portfolio_weights': portfolio_weights,
                'analysis_date': datetime.now().isoformat(),
                'individual_analyses': portfolio_data,
                'portfolio_metrics': portfolio_analysis,
                'recommendations': self._generate_portfolio_recommendations(portfolio_analysis)
            }

            # 导出报告
            if export_formats:
                await self._export_reports(portfolio_report, export_formats, output_path, "portfolio")

            self.logger.info("投资组合分析完成")
            return portfolio_report

        except Exception as e:
            self.logger.error(f"投资组合分析失败: {str(e)}")
            raise

    async def _export_reports(self, report_data: Dict, formats: List[str],
                             output_path: Path, base_name: str) -> None:
        """导出报告"""
        for format_name in formats:
            try:
                format_enum = ExportFormat(format_name.lower())
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{base_name}_analysis_{timestamp}.{format_name.lower()}"
                file_path = output_path / filename

                config = ExportConfig(
                    format=format_enum,
                    output_path=str(file_path),
                    include_charts=True,
                    include_raw_data=False
                )

                await self.export_manager.export_report(report_data, config)
                self.logger.info(f"报告已导出: {file_path}")

            except ValueError as e:
                self.logger.error(f"不支持的导出格式: {format_name}")
            except Exception as e:
                self.logger.error(f"导出失败 {format_name}: {str(e)}")

    async def _analyze_portfolio_metrics(self, portfolio_data: Dict,
                                       weights: Dict[str, float] = None) -> Dict:
        """分析投资组合指标"""
        # 简化的投资组合分析逻辑
        total_return = 0
        total_risk = 0
        weights = weights or {symbol: 1/len(portfolio_data) for symbol in portfolio_data}

        for symbol, data in portfolio_data.items():
            weight = weights.get(symbol, 0)
            if 'financial_metrics' in data:
                # 简化的收益率计算
                return_rate = data['financial_metrics'].get('roe', 0) / 100
                total_return += weight * return_rate

            if 'risk_metrics' in data:
                # 简化的风险计算
                volatility = data['risk_metrics'].get('volatility', 0)
                total_risk += weight * volatility

        return {
            'expected_return': total_return,
            'portfolio_risk': total_risk,
            'sharpe_ratio': total_return / total_risk if total_risk > 0 else 0,
            'diversification_score': len(portfolio_data) / 10  # 简化的分散化评分
        }

    def _generate_summary(self, financial_metrics: Dict, risk_metrics: Dict) -> Dict:
        """生成分析摘要"""
        try:
            # 简化的摘要生成逻辑
            roe = financial_metrics.get('roe', 0)
            roa = financial_metrics.get('roa', 0)
            debt_ratio = financial_metrics.get('debt_ratio', 0)
            volatility = risk_metrics.get('volatility', 0)

            # 综合评分
            score = 0
            if roe > 15: score += 25
            if roa > 10: score += 25
            if debt_ratio < 60: score += 25
            if volatility < 30: score += 25

            return {
                'overview': f"ROE: {roe:.1f}%, ROA: {roa:.1f}%, 资产负债率: {debt_ratio:.1f}%, 波动率: {volatility:.1f}%",
                'score': score,
                'rating': self._get_rating(score)
            }

        except Exception as e:
            self.logger.error(f"摘要生成失败: {str(e)}")
            return {'overview': '无法生成摘要', 'score': 0, 'rating': 'N/A'}

    def _generate_recommendations(self, financial_metrics: Dict, risk_metrics: Dict) -> List[str]:
        """生成投资建议"""
        recommendations = []

        try:
            roe = financial_metrics.get('roe', 0)
            debt_ratio = financial_metrics.get('debt_ratio', 0)
            volatility = risk_metrics.get('volatility', 0)

            if roe > 20:
                recommendations.append("盈利能力优秀，具备长期投资价值")
            elif roe > 10:
                recommendations.append("盈利能力良好，建议关注")
            else:
                recommendations.append("盈利能力有待改善，建议谨慎投资")

            if debt_ratio > 70:
                recommendations.append("负债率较高，关注财务风险")
            elif debt_ratio > 50:
                recommendations.append("负债率适中，关注偿债能力")

            if volatility > 40:
                recommendations.append("波动率较高，适合风险承受能力强的投资者")
            elif volatility > 25:
                recommendations.append("波动率适中，适合平衡型投资者")
            else:
                recommendations.append("波动率较低，适合保守型投资者")

        except Exception as e:
            self.logger.error(f"建议生成失败: {str(e)}")
            recommendations.append("无法生成具体建议，请结合其他指标分析")

        return recommendations

    def _generate_portfolio_recommendations(self, portfolio_analysis: Dict) -> List[str]:
        """生成投资组合建议"""
        recommendations = []

        try:
            expected_return = portfolio_analysis.get('expected_return', 0)
            portfolio_risk = portfolio_analysis.get('portfolio_risk', 0)
            sharpe_ratio = portfolio_analysis.get('sharpe_ratio', 0)

            if sharpe_ratio > 1:
                recommendations.append("投资组合风险调整收益优秀，配置合理")
            elif sharpe_ratio > 0.5:
                recommendations.append("投资组合风险调整收益良好，可适度优化")
            else:
                recommendations.append("投资组合风险调整收益有待改善，建议重新配置")

            if expected_return > 0.15:
                recommendations.append("预期收益率较高，适合成长型投资策略")
            elif expected_return > 0.08:
                recommendations.append("预期收益率适中，适合平衡型投资策略")
            else:
                recommendations.append("预期收益率较低，建议优化资产配置")

        except Exception as e:
            self.logger.error(f"投资组合建议生成失败: {str(e)}")
            recommendations.append("无法生成具体建议，请重新评估投资组合")

        return recommendations

    def _get_rating(self, score: float) -> str:
        """获取评级"""
        if score >= 80:
            return "优秀"
        elif score >= 60:
            return "良好"
        elif score >= 40:
            return "一般"
        else:
            return "较差"

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.system_monitor:
                self.system_monitor.stop_monitoring()

            if self.cache_manager:
                await self.cache_manager.close()

            self.logger.info("资源清理完成")

        except Exception as e:
            if self.logger:
                self.logger.error(f"资源清理失败: {str(e)}")
            else:
                print(f"资源清理失败: {str(e)}")


def create_cli_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        description="AutoGen金融分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python -m src.main analyze AAPL --format html pdf
  python -m src.main portfolio AAPL MSFT GOOG --weights AAPL=0.5 MSFT=0.3 GOOG=0.2
  python -m src.main analyze AAPL --type quick --output ./results
        """
    )

    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )

    parser.add_argument(
        "--output", "-o",
        default="output",
        help="输出目录 (默认: output)"
    )

    parser.add_argument(
        "--format", "-f",
        nargs="+",
        choices=["html", "pdf", "excel", "csv", "json", "markdown"],
        default=["html"],
        help="导出格式 (默认: html)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 分析单个公司
    analyze_parser = subparsers.add_parser("analyze", help="分析单个公司")
    analyze_parser.add_argument("symbol", help="股票代码")
    analyze_parser.add_argument(
        "--type", "-t",
        choices=["quick", "comprehensive", "detailed"],
        default="comprehensive",
        help="分析类型 (默认: comprehensive)"
    )

    # 分析投资组合
    portfolio_parser = subparsers.add_parser("portfolio", help="分析投资组合")
    portfolio_parser.add_argument("symbols", nargs="+", help="股票代码列表")
    portfolio_parser.add_argument(
        "--weights",
        help="投资组合权重，格式: AAPL=0.5,MSFT=0.3,GOOG=0.2"
    )

    # 交互模式
    interactive_parser = subparsers.add_parser("interactive", help="交互模式")

    return parser


def parse_weights(weights_str: str) -> Dict[str, float]:
    """解析权重字符串"""
    weights = {}
    if not weights_str:
        return weights

    for weight_pair in weights_str.split(","):
        try:
            symbol, weight = weight_pair.split("=")
            weights[symbol.strip()] = float(weight.strip())
        except ValueError:
            raise ValueError(f"权重格式错误: {weight_pair}")

    return weights


async def main():
    """主函数"""
    parser = create_cli_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # 初始化系统
        system = AutoGenFinancialAnalysisSystem(args.config)
        await system.initialize()

        if args.command == "analyze":
            # 分析单个公司
            result = await system.analyze_company(
                symbol=args.symbol,
                analysis_type=args.type,
                export_formats=args.format,
                output_dir=args.output
            )
            print(f"分析完成: {args.symbol}")
            print(f"综合评分: {result.get('summary', {}).get('score', 'N/A')}")
            print(f"评级: {result.get('summary', {}).get('rating', 'N/A')}")

        elif args.command == "portfolio":
            # 分析投资组合
            weights = parse_weights(args.weights) if args.weights else None
            result = await system.analyze_portfolio(
                symbols=args.symbols,
                portfolio_weights=weights,
                export_formats=args.format,
                output_dir=args.output
            )
            print(f"投资组合分析完成: {', '.join(args.symbols)}")
            portfolio_metrics = result.get('portfolio_metrics', {})
            print(f"预期收益率: {portfolio_metrics.get('expected_return', 0):.2%}")
            print(f"投资组合风险: {portfolio_metrics.get('portfolio_risk', 0):.2%}")
            print(f"夏普比率: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")

        elif args.command == "interactive":
            # 交互模式
            await interactive_mode(system)

    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"执行失败: {str(e)}")
        sys.exit(1)
    finally:
        if 'system' in locals():
            await system.cleanup()


async def interactive_mode(system: AutoGenFinancialAnalysisSystem):
    """交互模式"""
    print("AutoGen金融分析系统 - 交互模式")
    print("输入 'help' 查看帮助，输入 'quit' 退出")

    while True:
        try:
            command = input("\n> ").strip()

            if command.lower() in ['quit', 'exit', 'q']:
                break
            elif command.lower() == 'help':
                print_interactive_help()
            elif command.startswith('analyze'):
                parts = command.split()
                if len(parts) >= 2:
                    symbol = parts[1]
                    result = await system.analyze_company(symbol)
                    print(f"\n{symbol} 分析结果:")
                    print(f"综合评分: {result.get('summary', {}).get('score', 'N/A')}")
            elif command.startswith('portfolio'):
                parts = command.split()
                if len(parts) >= 2:
                    symbols = parts[1:]
                    result = await system.analyze_portfolio(symbols)
                    print(f"\n投资组合分析完成: {', '.join(symbols)}")
            else:
                print("未知命令，输入 'help' 查看帮助")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"执行错误: {str(e)}")


def print_interactive_help():
    """打印交互模式帮助"""
    print("""
可用命令:
  analyze <symbol>         - 分析单个公司 (例如: analyze AAPL)
  portfolio <symbols...>   - 分析投资组合 (例如: portfolio AAPL MSFT GOOG)
  help                    - 显示帮助
  quit/exit/q             - 退出程序
    """)


if __name__ == "__main__":
    asyncio.run(main())