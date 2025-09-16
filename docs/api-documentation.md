# AutoGen金融分析系统API文档

## 1. 概述

AutoGen金融分析系统提供了功能丰富的RESTful API接口，方便与其他系统集成和自动化操作。本文档详细描述了系统提供的所有API端点、请求参数、响应格式和使用示例。

## 2. 基础信息

### 2.1 API版本

当前API版本：v1

### 2.2 基础URL

```
http://localhost:8000/api/v1/
```

### 2.3 认证方式

系统API使用JWT（JSON Web Token）进行身份认证：

1. 获取令牌：通过`/auth/token`端点获取JWT令牌
2. 在后续请求的Authorization头中包含令牌：`Bearer {token}`

### 2.4 响应格式

所有API响应都使用JSON格式，包含以下基本字段：

```json
{
  "status": "success" | "error",
  "data": { ... },  // 成功响应的数据
  "error": "..."   // 错误信息（仅在status为error时存在）
}
```

### 2.5 错误码

| 错误码 | 描述 | HTTP状态码 |
|-------|------|------------|
| 40001 | 请求参数错误 | 400 |
| 40101 | 未授权访问 | 401 |
| 40301 | 权限不足 | 403 |
| 40401 | 资源不存在 | 404 |
| 50001 | 服务器内部错误 | 500 |
| 50002 | 数据分析失败 | 500 |
| 50003 | 数据源连接失败 | 500 |

## 3. 认证API

### 3.1 获取认证令牌

**请求**: POST /auth/token

**请求参数**: 

```json
{
  "username": "string",
  "password": "string"
}
```

**响应**: 

```json
{
  "status": "success",
  "data": {
    "access_token": "string",
    "token_type": "bearer",
    "expires_in": 3600
  }
}
```

**示例**: 

```bash
curl -X POST "http://localhost:8000/api/v1/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'
```

### 3.2 刷新令牌

**请求**: POST /auth/refresh

**请求头**: Authorization: Bearer {refresh_token}

**响应**: 

```json
{
  "status": "success",
  "data": {
    "access_token": "string",
    "token_type": "bearer",
    "expires_in": 3600
  }
}
```

**示例**: 

```bash
curl -X POST "http://localhost:8000/api/v1/auth/refresh" \
  -H "Authorization: Bearer {refresh_token}"
```

### 3.3 验证令牌

**请求**: GET /auth/verify

**请求头**: Authorization: Bearer {token}

**响应**: 

```json
{
  "status": "success",
  "data": {
    "valid": true,
    "username": "string",
    "expires_at": "2023-12-31T23:59:59Z"
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/auth/verify" \
  -H "Authorization: Bearer {token}"
```

## 4. 分析API

### 4.1 创建分析任务

**请求**: POST /analysis

**请求头**: Authorization: Bearer {token}

**请求参数**: 

```json
{
  "symbols": ["string"],  // 股票代码列表
  "analysis_type": "quick" | "comprehensive" | "detailed",  // 分析类型
  "export_formats": ["html" | "pdf" | "json" | "csv"],  // 导出格式
  "portfolio_weights": {  // 投资组合权重（可选，仅对多个股票有效）
    "AAPL": 0.5,
    "MSFT": 0.3,
    "GOOG": 0.2
  }
}
```

**响应**: 

```json
{
  "status": "success",
  "data": {
    "task_id": "string",
    "status": "pending",
    "created_at": "2023-12-01T12:00:00Z",
    "estimated_completion": "2023-12-01T12:05:00Z"
  }
}
```

**示例**: 

```bash
curl -X POST "http://localhost:8000/api/v1/analysis" \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "analysis_type": "comprehensive",
    "export_formats": ["html", "pdf"]
  }'
```

### 4.2 获取分析任务状态

**请求**: GET /analysis/{task_id}

**请求头**: Authorization: Bearer {token}

**路径参数**: 
- `task_id`: 任务ID

**响应**: 

```json
{
  "status": "success",
  "data": {
    "task_id": "string",
    "status": "pending" | "processing" | "completed" | "failed",
    "created_at": "2023-12-01T12:00:00Z",
    "updated_at": "2023-12-01T12:03:00Z",
    "progress": 60,  // 进度百分比
    "error_message": "string",  // 仅在状态为failed时存在
    "results": {  // 仅在状态为completed时存在
      "summary": {},
      "financial_metrics": {},
      "risk_metrics": {},
      "reports": [
        {"format": "html", "url": "http://localhost:8000/reports/xxx.html"},
        {"format": "pdf", "url": "http://localhost:8000/reports/xxx.pdf"}
      ]
    }
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/analysis/{task_id}" \
  -H "Authorization: Bearer {token}"
```

### 4.3 获取分析结果

**请求**: GET /analysis/{task_id}/results

**请求头**: Authorization: Bearer {token}

**路径参数**: 
- `task_id`: 任务ID

**查询参数**: 
- `format`: 结果格式，可选值：`json`（默认）、`csv`

**响应**: 

```json
{
  "status": "success",
  "data": {
    "summary": {
      "score": 85,
      "rating": "买入",
      "overview": "..."
    },
    "financial_metrics": {
      "roe": 20.5,
      "roa": 10.2,
      "profit_margin": 15.8,
      "debt_ratio": 45.3,
      // 更多财务指标
    },
    "risk_metrics": {
      "volatility": 22.5,
      "beta": 1.15,
      "var_95": 3.2,
      // 更多风险指标
    },
    "recommendations": [
      {"type": "投资建议", "content": "..."},
      {"type": "风险提示", "content": "..."}
    ]
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/analysis/{task_id}/results" \
  -H "Authorization: Bearer {token}"
```

### 4.4 取消分析任务

**请求**: DELETE /analysis/{task_id}

**请求头**: Authorization: Bearer {token}

**路径参数**: 
- `task_id`: 任务ID

**响应**: 

```json
{
  "status": "success",
  "data": {
    "task_id": "string",
    "status": "cancelled",
    "cancelled_at": "2023-12-01T12:02:00Z"
  }
}
```

**示例**: 

```bash
curl -X DELETE "http://localhost:8000/api/v1/analysis/{task_id}" \
  -H "Authorization: Bearer {token}"
```

## 5. 数据API

### 5.1 获取股票基本信息

**请求**: GET /data/stocks/{symbol}

**请求头**: Authorization: Bearer {token}

**路径参数**: 
- `symbol`: 股票代码

**响应**: 

```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "exchange": "NASDAQ",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "country": "United States",
    "market_cap": 2.8e12,
    "currency": "USD",
    "updated_at": "2023-12-01T12:00:00Z"
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/data/stocks/AAPL" \
  -H "Authorization: Bearer {token}"
```

### 5.2 获取历史价格数据

**请求**: GET /data/stocks/{symbol}/prices

**请求头**: Authorization: Bearer {token}

**路径参数**: 
- `symbol`: 股票代码

**查询参数**: 
- `start_date`: 开始日期，格式：YYYY-MM-DD
- `end_date`: 结束日期，格式：YYYY-MM-DD
- `interval`: 时间间隔，可选值：`1d`（默认）、`1wk`、`1mo`

**响应**: 

```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "interval": "1d",
    "prices": [
      {"date": "2023-12-01", "open": 188.0, "high": 190.5, "low": 187.2, "close": 189.8, "volume": 45000000},
      {"date": "2023-11-30", "open": 186.5, "high": 188.2, "low": 186.1, "close": 187.9, "volume": 38000000},
      // 更多价格数据
    ],
    "updated_at": "2023-12-01T12:00:00Z"
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/data/stocks/AAPL/prices?start_date=2023-01-01&end_date=2023-12-01" \
  -H "Authorization: Bearer {token}"
```

### 5.3 获取财务报表数据

**请求**: GET /data/stocks/{symbol}/financials

**请求头**: Authorization: Bearer {token}

**路径参数**: 
- `symbol`: 股票代码

**查询参数**: 
- `report_type`: 报表类型，可选值：`income`（利润表，默认）、`balance`（资产负债表）、`cash_flow`（现金流量表）
- `period`: 报告周期，可选值：`annual`（年报，默认）、`quarterly`（季报）

**响应**: 

```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "report_type": "income",
    "period": "annual",
    "reports": [
      {
        "fiscal_year": 2023,
        "fiscal_quarter": null,
        "revenue": 383.285e9,
        "net_income": 94.8 billion,
        "eps": 6.13,
        // 更多财务数据
      },
      // 更多报告
    ],
    "updated_at": "2023-12-01T12:00:00Z"
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/data/stocks/AAPL/financials?report_type=balance&period=quarterly" \
  -H "Authorization: Bearer {token}"
```

### 5.4 获取市场指标数据

**请求**: GET /data/market/indices

**请求头**: Authorization: Bearer {token}

**查询参数**: 
- `indices`: 指数代码列表，例如：`^GSPC,^DJI,^IXIC`
- `start_date`: 开始日期，格式：YYYY-MM-DD
- `end_date`: 结束日期，格式：YYYY-MM-DD

**响应**: 

```json
{
  "status": "success",
  "data": {
    "indices": [
      {
        "symbol": "^GSPC",
        "name": "S&P 500",
        "prices": [
          {"date": "2023-12-01", "close": 4567.89},
          // 更多价格数据
        ]
      },
      // 更多指数
    ],
    "updated_at": "2023-12-01T12:00:00Z"
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/data/market/indices?indices=^GSPC,^DJI" \
  -H "Authorization: Bearer {token}"
```

## 6. 报告API

### 6.1 获取报告列表

**请求**: GET /reports

**请求头**: Authorization: Bearer {token}

**查询参数**: 
- `symbol`: 股票代码（可选，用于过滤）
- `type`: 报告类型（可选，例如：`financial`, `risk`, `portfolio`）
- `start_date`: 开始日期，格式：YYYY-MM-DD
- `end_date`: 结束日期，格式：YYYY-MM-DD
- `limit`: 返回的最大数量（默认：10）
- `offset`: 偏移量（默认：0）

**响应**: 

```json
{
  "status": "success",
  "data": {
    "reports": [
      {
        "id": "string",
        "symbol": "AAPL",
        "type": "financial",
        "analysis_type": "comprehensive",
        "created_at": "2023-12-01T12:00:00Z",
        "formats": ["html", "pdf"],
        "url": "http://localhost:8000/reports/xxx.html"
      },
      // 更多报告
    ],
    "total": 42,
    "limit": 10,
    "offset": 0
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/reports?symbol=AAPL&limit=20" \
  -H "Authorization: Bearer {token}"
```

### 6.2 获取报告详情

**请求**: GET /reports/{report_id}

**请求头**: Authorization: Bearer {token}

**路径参数**: 
- `report_id`: 报告ID

**响应**: 

```json
{
  "status": "success",
  "data": {
    "id": "string",
    "symbol": "AAPL",
    "type": "financial",
    "analysis_type": "comprehensive",
    "created_at": "2023-12-01T12:00:00Z",
    "summary": {
      "score": 85,
      "rating": "买入",
      "overview": "..."
    },
    "formats": [
      {"type": "html", "url": "http://localhost:8000/reports/xxx.html"},
      {"type": "pdf", "url": "http://localhost:8000/reports/xxx.pdf"},
      {"type": "json", "url": "http://localhost:8000/reports/xxx.json"}
    ]
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/reports/{report_id}" \
  -H "Authorization: Bearer {token}"
```

### 6.3 删除报告

**请求**: DELETE /reports/{report_id}

**请求头**: Authorization: Bearer {token}

**路径参数**: 
- `report_id`: 报告ID

**响应**: 

```json
{
  "status": "success",
  "data": {
    "id": "string",
    "deleted": true
  }
}
```

**示例**: 

```bash
curl -X DELETE "http://localhost:8000/api/v1/reports/{report_id}" \
  -H "Authorization: Bearer {token}"
```

## 7. 系统管理API

### 7.1 获取系统状态

**请求**: GET /system/status

**请求头**: Authorization: Bearer {token}

**响应**: 

```json
{
  "status": "success",
  "data": {
    "version": "1.0.0",
    "status": "running",
    "uptime": "2 days, 4 hours, 30 minutes",
    "memory_usage": "780MB / 16GB",
    "cpu_usage": "15%",
    "active_tasks": 5,
    "database": {
      "status": "connected",
      "connections": 12
    },
    "redis": {
      "status": "connected",
      "memory_usage": "240MB",
      "hit_rate": "92%"
    }
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/system/status" \
  -H "Authorization: Bearer {token}"
```

### 7.2 获取系统配置

**请求**: GET /system/config

**请求头**: Authorization: Bearer {token}

**响应**: 

```json
{
  "status": "success",
  "data": {
    "autogen": {
      "gpt_model": "gpt-4",
      "temperature": 0.7,
      "max_tokens": 4000
    },
    "data_sources": {
      "yahoo_finance": {
        "timeout": 30,
        "retry_count": 3
      },
      "alpha_vantage": {
        "api_key": "******",
        "calls_per_minute": 5
      }
    },
    "cache": {
      "enabled": true,
      "ttl": 3600
    },
    "security": {
      "jwt_expiration": 3600,
      "rate_limit": "100/hour"
    }
  }
}
```

**示例**: 

```bash
curl -X GET "http://localhost:8000/api/v1/system/config" \
  -H "Authorization: Bearer {token}"
```

### 7.3 更新系统配置

**请求**: PUT /system/config

**请求头**: Authorization: Bearer {token}

**请求参数**: 

```json
{
  "section": "string",  // 配置部分，例如："autogen", "data_sources", "cache"
  "config": {
    // 要更新的配置项
  }
}
```

**响应**: 

```json
{
  "status": "success",
  "data": {
    "updated": true,
    "restart_required": false
  }
}
```

**示例**: 

```bash
curl -X PUT "http://localhost:8000/api/v1/system/config" \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "section": "cache",
    "config": {
      "enabled": true,
      "ttl": 7200
    }
  }'
```

## 8. WebSocket API

系统还提供了WebSocket API，用于获取实时更新和推送通知。

### 8.1 连接WebSocket

**URL**: `ws://localhost:8000/ws`

**认证**: 连接时需要提供JWT令牌作为查询参数：`ws://localhost:8000/ws?token={token}`

### 8.2 接收任务更新

连接建立后，客户端会收到分析任务的实时更新：

```json
{
  "type": "task_update",
  "data": {
    "task_id": "string",
    "status": "processing",
    "progress": 60,
    "message": "正在进行风险评估..."
  }
}
```

### 8.3 接收系统通知

客户端还会收到系统级别的通知：

```json
{
  "type": "system_notification",
  "data": {
    "level": "info" | "warning" | "error",
    "title": "数据源连接恢复",
    "message": "Yahoo Finance数据源连接已恢复正常",
    "timestamp": "2023-12-01T12:05:00Z"
  }
}
```

### 8.4 JavaScript示例

```javascript
const token = "your_jwt_token";
const ws = new WebSocket(`ws://localhost:8000/ws?token=${token}`);

ws.onopen = function(event) {
  console.log('WebSocket连接已建立');
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  if (data.type === 'task_update') {
    console.log('任务更新:', data.data);
    // 处理任务更新
  } else if (data.type === 'system_notification') {
    console.log('系统通知:', data.data);
    // 处理系统通知
  }
};

ws.onerror = function(error) {
  console.error('WebSocket错误:', error);
};

ws.onclose = function(event) {
  console.log('WebSocket连接已关闭');
};
```

## 9. API使用最佳实践

### 9.1 认证与安全

- 始终使用HTTPS协议（生产环境）
- 妥善保管JWT令牌，不要在客户端代码中硬编码
- 设置合理的令牌过期时间
- 定期刷新令牌
- 实施适当的访问控制和权限检查

### 9.2 性能优化

- 使用批量操作减少API调用次数
- 利用缓存减少重复请求
- 合理设置请求超时时间
- 实现重试机制处理临时错误
- 监控API响应时间并进行优化

### 9.3 错误处理

- 检查并处理所有API错误响应
- 实现适当的重试逻辑
- 记录详细的错误日志
- 向用户提供友好的错误信息
- 区分临时错误和永久错误

### 9.4 分页与过滤

- 对大型数据集使用分页
- 使用过滤参数减少数据传输量
- 按需求选择适当的数据字段
- 实现增量同步机制减少数据传输

## 10. 常见问题

**Q: 如何处理API速率限制？**
A: 系统实现了速率限制机制，建议在应用中实现重试逻辑，并尊重API的速率限制。

**Q: 如何获取大量历史数据？**
A: 对于大量历史数据，建议使用较小的时间范围分批获取，或使用系统提供的数据导出功能。

**Q: API返回的金融数据与实际市场数据有差异怎么办？**
A: 数据可能存在延迟，请检查数据的更新时间。如果发现持续的差异，请联系技术支持。

**Q: 如何调试API请求？**
A: 可以使用Postman、curl等工具测试API请求，检查请求参数和响应结果。

**Q: API版本更新会影响现有应用吗？**
A: 我们会尽力保持API的向后兼容性，但建议在新版本发布后及时测试和更新您的应用。

## 11. API变更日志

### v1.0.0 (2023-12-01)
- 初始版本发布
- 实现认证API
- 实现分析API
- 实现数据API
- 实现报告API
- 实现系统管理API
- 实现WebSocket API

---

**注意**: 本API仅供学习和研究使用，不构成投资建议。投资有风险，请谨慎决策。