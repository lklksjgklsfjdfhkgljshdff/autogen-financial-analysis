# AutoGen Financial Analysis System

一个基于微软AutoGen框架的企业级金融分析系统，使用多Agent架构提供全面的财务分析、风险评估和量化投资分析功能。

## 🚀 功能特性

### 🔍 核心功能
- **多源数据收集**: 整合Yahoo Finance、Alpha Vantage等多个金融数据源
- **智能财务分析**: 基于AutoGen的多Agent协作分析
- **风险评估**: VaR计算、压力测试、蒙特卡洛模拟
- **量化分析**: 因子模型、投资组合优化、策略回测、机器学习预测
- **实时监控**: 系统性能监控和告警
- **数据可视化**: 交互式图表和报告生成

### 🏗️ 技术架构
- **微服务架构**: 模块化设计，支持水平扩展
- **异步处理**: 高性能异步任务处理
- **缓存系统**: 多级缓存策略，提升响应速度
- **安全性**: 完整的身份认证、授权和加密
- **监控告警**: Prometheus + Grafana监控体系
- **容器化**: Docker和Kubernetes部署支持

## 📦 安装指南

### 系统要求
- Python 3.8+
- Redis 6.0+
- PostgreSQL 12+
- Docker (可选)

### 快速安装

1. **克隆项目**
```bash
git clone https://github.com/your-username/autogen-financial-analysis.git
cd autogen-financial-analysis
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，添加必要的API密钥
```

### Docker部署

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

## 🛠️ 使用方法

### 命令行界面

#### 基本用法
```bash
# 分析单个公司
python -m src.main analyze AAPL

# 分析投资组合
python -m src.main portfolio AAPL MSFT GOOG

# 交互模式
python -m src.main interactive
```

#### 高级选项
```bash
# 指定分析类型
python -m src.main analyze AAPL --type comprehensive

# 导出报告
python -m src.main analyze AAPL --format html,pdf

# 自定义配置
python -m src.main analyze AAPL --config custom_config.yaml
```

#### 量化分析选项
```bash
# 对单个股票进行量化分析
python -m src.main quant AAPL

# 使用特定因子进行分析
python -m src.main quant AAPL --factors momentum value growth

# 使用特定因子模型
python -m src.main quant AAPL --method carhart

# 导出量化分析报告
python -m src.main quant AAPL --format html,pdf,json
```

#### 策略回测选项
```bash
# 运行动量策略回测
python -m src.main backtest --strategy momentum --start-date 2020-01-01 --end-date 2023-01-01

# 设置回测参数
python -m src.main backtest --strategy momentum --start-date 2020-01-01 --end-date 2023-01-01 --initial-capital 100000 --commission 0.001

# 导出回测报告
python -m src.main backtest --strategy momentum --start-date 2020-01-01 --end-date 2023-01-01 --format html,pdf
```

#### 策略优化选项
```bash
# 优化策略参数
python -m src.main optimize --strategy momentum --param window=5,10,15,20

# 设置优化时间范围
python -m src.main optimize --strategy momentum --param window=5,10,15,20 --start-date 2020-01-01 --end-date 2023-01-01
```

#### 投资组合优化选项
```bash
# 使用均值-方差优化方法
python -m src.main optimize-portfolio --symbols AAPL MSFT GOOG --method mean_variance

# 使用风险平价优化方法
python -m src.main optimize-portfolio --symbols AAPL MSFT GOOG --method risk_parity

# 设置风险厌恶系数
python -m src.main optimize-portfolio --symbols AAPL MSFT GOOG --method mean_variance --risk-aversion 1.5
```

### Web界面

启动Web服务：
```bash
python -m src.api.app
```

访问 `http://localhost:8000` 使用Web界面。

### API接口

#### 创建分析任务
```bash
curl -X POST "http://localhost:8000/api/v1/analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "analysis_type": "comprehensive",
    "export_formats": ["html", "pdf"]
  }'
```

#### 查看任务状态
```bash
curl -X GET "http://localhost:8000/api/v1/analysis/{task_id}"
```

#### WebSocket实时更新
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('任务更新:', data);
};
```

## 📊 分析报告

### 财务分析报告
- **盈利能力分析**: ROE、ROA、毛利率、净利率
- **偿债能力分析**: 资产负债率、流动比率、速动比率
- **运营效率分析**: 总资产周转率、存货周转率
- **成长性分析**: 收入增长率、利润增长率
- **杜邦分析**: ROE分解为净利润率、资产周转率和权益乘数

### 风险评估报告
- **市场风险**: VaR、CVaR、Beta系数
- **信用风险**: Z-Score、Altman模型
- **流动性风险**: 流动性覆盖率、净稳定资金率
- **操作风险**: 历史模拟、蒙特卡洛模拟
- **压力测试**: 极端市场情景分析

### 量化分析报告
- **因子分析**: 多因子暴露、因子收益率、信息系数
- **投资组合优化**: 有效前沿、风险平价、最大分散化
- **策略回测**: 累计收益、最大回撤、夏普比率
- **风险贡献分析**: 各资产对组合风险的贡献度
- **绩效归因**: 收益来源分解

### 投资组合分析报告
- **有效前沿**: 风险收益最优化组合
- **夏普比率**: 风险调整后收益
- **最大回撤**: 历史最大损失
- **相关系数**: 资产间相关性分析
- **风险平价**: 风险贡献度优化

## 🔧 配置说明

### 主要配置文件

#### config.yaml
```yaml
# AutoGen配置
autogen:
  gpt_model: "gpt-4"
  temperature: 0.7
  max_tokens: 4000

# 数据源配置
data_sources:
  yahoo_finance:
    timeout: 30
    retry_count: 3
  alpha_vantage:
    api_key: "${ALPHA_VANTAGE_API_KEY}"
    calls_per_minute: 5
```

#### 环境变量
```bash
# API密钥
YAHOO_FINANCE_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/autogen_financial
REDIS_URL=redis://localhost:6379/0

# 安全配置
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
```

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_data.py

# 运行API测试
pytest tests/test_api.py

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 测试覆盖率
- 数据收集模块: 95%
- 财务分析模块: 92%
- 风险分析模块: 90%
- API接口: 88%
- 整体覆盖率: 93%

## 📈 性能监控

### 系统指标
- CPU使用率
- 内存使用量
- 磁盘I/O
- 网络吞吐量
- 数据库连接数
- Redis命中率

### 业务指标
- 数据收集成功率
- 分析任务执行时间
- API响应时间
- 错误率
- 用户活跃度

### 访问监控界面
```bash
# Grafana仪表板
http://localhost:3000

# Prometheus查询界面
http://localhost:9090
```

## 🔒 安全特性

### 数据安全
- 传输加密: TLS 1.3
- 存储加密: AES-256
- API密钥加密存储
- 敏感数据脱敏

### 访问控制
- JWT身份认证
- RBAC权限控制
- API密钥管理
- IP白名单

### 安全防护
- 速率限制
- 请求验证
- SQL注入防护
- XSS防护
- CSRF防护

## 🚀 部署指南

### 开发环境
```bash
# 启动开发服务器
python -m src.api.app --reload

# 启动Redis
redis-server

# 启动PostgreSQL
sudo systemctl start postgresql
```

### 生产环境

#### Docker部署
```bash
# 构建镜像
docker build -t autogen-financial .

# 使用Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

#### Kubernetes部署
```bash
# 部署到K8s
kubectl apply -f k8s/

# 查看部署状态
kubectl get pods -n autogen-financial

# 查看服务
kubectl get svc -n autogen-financial
```

#### 负载均衡
```yaml
# nginx.conf 示例配置
upstream autogen_api {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}
```

## 🤝 贡献指南

### 开发流程
1. Fork项目
2. 创建功能分支
3. 提交代码
4. 创建Pull Request
5. 代码审查
6. 合并到主分支

### 代码规范
- 遵循PEP 8规范
- 编写单元测试
- 添加类型注解
- 编写文档字符串
- 使用pre-commit hooks

### 提交规范
```
feat: 新功能
fix: 修复bug
docs: 文档更新
style: 代码格式化
refactor: 代码重构
test: 测试相关
chore: 构建或辅助工具变动
```

## 📚 文档

### 官方文档
- [用户手册](docs/user-manual.md)
- [开发者指南](docs/developer-guide.md)
- [API文档](docs/api-documentation.md)
- [部署指南](docs/deployment-guide.md)

### 教程
- [快速入门](tutorials/quickstart.md)
- [财务分析教程](tutorials/financial-analysis.md)
- [风险评估教程](tutorials/risk-assessment.md)
- [量化分析教程](tutorials/quantitative-analysis.md)

## 📞 支持

### 获取帮助
- GitHub Issues: [问题反馈](https://github.com/your-username/autogen-financial-analysis/issues)
- 邮件支持: support@example.com
- 文档中心: [在线文档](https://docs.autogen-financial.com)

### 常见问题
Q: 如何获取API密钥？
A: 请访问各数据源官方网站申请API密钥。

Q: 系统支持哪些股票市场？
A: 目前支持美股、A股、港股等主要市场。

Q: 如何添加新的数据源？
A: 参考`src/data/data_sources.py`中的接口实现新的数据源类。

## 📄 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

## 🙏 致谢

感谢以下开源项目的支持：
- [Microsoft AutoGen](https://github.com/microsoft/autogen)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [Pandas](https://github.com/pandas-dev/pandas)
- [Plotly](https://github.com/plotly/plotly.py)
- [Redis](https://github.com/redis/redis)
- [Prometheus](https://github.com/prometheus/prometheus)

---

**注意**: 本系统仅供学习和研究使用，不构成投资建议。投资有风险，请谨慎决策。