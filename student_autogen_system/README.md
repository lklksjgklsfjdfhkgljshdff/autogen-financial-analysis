# 学生级AutoGen金融分析系统

这是一个简化的AutoGen金融分析系统，专为学习和教学设计。系统基于Microsoft的AutoGen框架，可以自动执行股票数据收集、财务分析、风险评估和报告生成。

## 🎯 系统特色

- **简化设计**：相比企业级系统，这个版本更加简洁易懂
- **突出AutoGen核心**：重点展示智能体协作的工作流程
- **易于扩展**：模块化设计，便于学生修改和扩展
- **免费数据源**：使用Yahoo Finance作为数据源，无需付费API

## 📁 系统架构

```
student_autogen_system/
├── main_system.py               # 主系统文件
├── requirements.txt              # 依赖包列表
├── README.md                     # 说明文档
├── example_usage.py              # 使用示例
├── test_basic_functionality.py   # 基础功能测试
└── SYSTEM_REPORT.md              # 系统状态报告
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

系统需要OpenAI的API密钥来使用AutoGen功能：

```python
# 将你的API密钥替换 "your-openai-api-key"
api_key = "your-openai-api-key"
```

### 3. 基本使用

```python
from main_system import StudentAutoGenSystem

# 创建分析系统
system = StudentAutoGenSystem("your-openai-api-key")

# 分析股票
result = system.analyze_stock("AAPL")  # 苹果公司

# 查看结果
print(result["report"])
```

## 🧠 AutoGen核心概念

### 智能体类型

系统包含三个主要智能体：

1. **数据收集智能体** (`data_collector`)
   - 负责收集股票数据
   - 获取历史价格和财务报表
   - 验证数据完整性

2. **财务分析智能体** (`financial_analyst`)
   - 计算财务比率
   - 分析公司盈利能力
   - 识别风险点

3. **报告生成智能体** (`report_generator`)
   - 整理分析结果
   - 生成投资建议
   - 提供风险提示

### 工作流程

1. 数据收集 → 财务分析 → 风险评估 → 报告生成
2. 智能体之间通过AutoGen框架进行协作
3. 最终输出完整的分析报告

## 📊 分析功能

### 财务指标
- ROE（净资产收益率）
- ROA（总资产收益率）
- 资产负债率
- 毛利率
- 净利率
- 流动比率

### 风险指标
- 年化波动率
- VaR（风险价值）
- 最大回撤

## 🎓 学习要点

### 1. 理解AutoGen智能体
- 每个智能体都有特定的角色和功能
- 通过`system_message`定义智能体的行为
- 智能体之间可以进行对话和协作

### 2. 数据处理流程
- 学习如何从Yahoo Finance获取数据
- 理解数据清洗和预处理的重要性
- 掌握基本的财务比率计算方法

### 3. 系统集成
- 了解如何将不同的组件整合在一起
- 学习错误处理和日志记录
- 掌握基本的系统架构设计

## 🔧 扩展建议

### 初级扩展
1. **添加更多财务指标**
   ```python
   # 在BasicFinancialMetrics类中添加新字段
   pe_ratio: float = 0.0
   pb_ratio: float = 0.0
   ```

2. **增加更多股票分析**
   ```python
   # 分析多只股票
   symbols = ["AAPL", "GOOGL", "MSFT"]
   for symbol in symbols:
       result = system.analyze_stock(symbol)
   ```

### 中级扩展
1. **添加可视化功能**
   ```python
   import matplotlib.pyplot as plt

   def plot_price_trend(data):
       plt.figure(figsize=(10, 6))
       plt.plot(data['price_data']['Close'])
       plt.title('股价趋势')
       plt.show()
   ```

2. **增强智能体能力**
   ```python
   # 修改智能体的system_message，添加更多功能
   agents['financial_analyst'] = autogen.AssistantAgent(
       name="financial_analyst",
       system_message="""你是一个专业的财务分析师...
       4. 进行行业对比分析
       5. 预测未来财务表现
       """,
       llm_config=self.base_config
   )
   ```

### 高级扩展
1. **添加新的数据源**
   ```python
   class AlphaVantageCollector:
       def collect_data(self, symbol):
           # 实现Alpha Vantage数据收集
           pass
   ```

2. **实现机器学习预测**
   ```python
   from sklearn.ensemble import RandomForestRegressor

   def predict_stock_price(data):
       # 使用机器学习模型预测股价
       pass
   ```

## 🐛 常见问题

### Q1: API密钥如何获取？
A1: 访问OpenAI官网 (https://openai.com/) 注册账户并获取API密钥。

### Q2: 数据收集失败怎么办？
A2: 检查网络连接，确保股票代码正确，或尝试更换其他股票。

### Q3: 如何降低成本？
A3:
- 使用gpt-3.5-turbo而不是gpt-4
- 减少智能体的对话次数
- 缓存分析结果避免重复计算

### Q4: 可以分析A股吗？
A4: 可以使用其他数据源，如`tushare`或`akshare`，需要相应修改数据收集器。

## 📚 相关资源

- [AutoGen官方文档](https://microsoft.github.io/autogen/)
- [yfinance文档](https://pypi.org/project/yfinance/)
- [pandas教程](https://pandas.pydata.org/docs/)
- [财务分析基础](https://www.investopedia.com/terms/f/financial-analysis.asp)

## 🤝 贡献指南

欢迎学生和开发者贡献代码和建议：

1. Fork本项目
2. 创建功能分支
3. 提交修改
4. 发起Pull Request

## 📄 许可证

本项目仅供学习和研究使用。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 创建Issue
- 发送邮件到项目维护者

---

*祝您学习愉快！希望这个系统能帮助您更好地理解AutoGen和金融分析。* 🎉