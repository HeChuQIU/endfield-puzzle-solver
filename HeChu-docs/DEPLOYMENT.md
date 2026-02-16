# 服务器部署指南

本文档描述如何使用 Docker + Caddy 部署后端 API 服务，并配置自动 HTTPS。

## 📋 前置要求

### 服务器环境
- Docker Engine 20.10+
- Docker Compose 2.0+
- 开放端口: 80 (HTTP), 443 (HTTPS)

### 域名配置
- DNS A 记录指向服务器 IP
- 等待 DNS 生效（通常 5-30 分钟）

验证 DNS：
```bash
nslookup hechuqiu.net
# 应该返回服务器 IP
```

## 🚀 部署步骤

### 1. 安装 Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 重新登录以应用权限
```

### 2. 上传配置文件

将以下文件上传到服务器目录（如 `/opt/endfield-puzzle-solver/`）：

```
/opt/endfield-puzzle-solver/
├── docker-compose.yml
└── Caddyfile
```

### 3. 启动服务

```bash
cd /opt/endfield-puzzle-solver/
docker-compose up -d
```

### 4. 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 只查看 Caddy 日志（包含证书申请过程）
docker-compose logs -f caddy

# 只查看 API 日志
docker-compose logs -f api
```

### 5. 验证部署

```bash
# 检查服务状态
docker-compose ps

# 测试 API
curl https://hechuqiu.net/api/health

# 预期返回
# {"status":"ok","timestamp":"..."}
```

## 🔧 配置文件说明

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    image: hechuqiu/endfield-puzzle-solver-api:latest
    restart: unless-stopped
    networks:
      - app-network

  caddy:
    image: caddy:latest
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
      - "443:443/udp"  # HTTP/3
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    networks:
      - app-network
    depends_on:
      - api

networks:
  app-network:

volumes:
  caddy_data:    # Caddy 数据（包含证书）
  caddy_config:  # Caddy 配置
```

### Caddyfile

```
hechuqiu.net {
    reverse_proxy api:8080
    
    # CORS 配置
    header {
        Access-Control-Allow-Origin *
        Access-Control-Allow-Methods "GET, POST, OPTIONS"
        Access-Control-Allow-Headers *
    }
    
    # 日志
    log {
        output stdout
        format console
    }
}
```

**说明**：
- `reverse_proxy api:8080`: 反向代理到 API 容器
- CORS 配置允许前端跨域访问
- Caddy 自动申请 Let's Encrypt 证书
- 证书保存在 Docker volume `caddy_data` 中

## 🔄 服务管理

### 停止服务
```bash
docker-compose down
```

### 重启服务
```bash
docker-compose restart
```

### 更新服务
```bash
# 拉取最新镜像
docker-compose pull

# 重启容器
docker-compose up -d
```

### 查看证书状态
```bash
docker-compose exec caddy caddy list-certificates
```

## 🔍 故障排查

### 证书申请失败

**症状**: 日志显示 `failed to obtain certificate`

**原因**:
1. DNS 未生效或配置错误
2. 端口 80/443 未开放
3. 防火墙阻止

**解决**:
```bash
# 检查 DNS
nslookup hechuqiu.net

# 检查端口
sudo netstat -tlnp | grep -E ':(80|443)'

# 检查防火墙（Ubuntu/Debian）
sudo ufw status
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

### API 无响应

**检查 API 容器状态**:
```bash
docker-compose logs api
docker-compose exec api curl http://localhost:8080/api/health
```

### CORS 错误

如果前端报 CORS 错误，检查 Caddyfile 中的 CORS 配置是否正确。

## 🔐 安全建议

### 1. 限制 API 访问（可选）

如果只允许特定域名访问，修改 Caddyfile：

```
hechuqiu.net {
    reverse_proxy api:8080
    
    header {
        Access-Control-Allow-Origin "https://hechuqiu.github.io"
        Access-Control-Allow-Methods "GET, POST, OPTIONS"
        Access-Control-Allow-Headers *
    }
}
```

### 2. 启用速率限制（可选）

安装 Caddy 速率限制插件：

```
hechuqiu.net {
    rate_limit {
        zone dynamic {
            key {remote_host}
            events 100
            window 1m
        }
    }
    
    reverse_proxy api:8080
}
```

### 3. 定期更新

```bash
# 每周检查更新
docker-compose pull
docker-compose up -d
```

## 📊 监控

### 资源使用

```bash
# 查看容器资源使用
docker stats

# 查看日志大小
docker-compose logs --tail=100 api
```

### 证书过期时间

```bash
docker-compose exec caddy caddy list-certificates
```

Caddy 会在证书到期前 30 天自动续期，无需手动操作。

## 🌐 前端配置

前端 `appsettings.json` 配置：

```json
{
  "App": {
    "ApiBaseUrl": "https://hechuqiu.net/"
  }
}
```

部署前端：
```bash
dotnet publish src/EndfieldPuzzleSolver.Web/EndfieldPuzzleSolver.Web.csproj -c Release -o publish/web
```

然后将 `publish/web/wwwroot/*` 部署到 GitHub Pages。

## 📝 备份

备份证书和配置：

```bash
# 备份 Caddy 数据卷
docker run --rm -v endfield-puzzle-solver_caddy_data:/data -v $(pwd):/backup alpine tar czf /backup/caddy-data-backup.tar.gz -C /data .

# 恢复
docker run --rm -v endfield-puzzle-solver_caddy_data:/data -v $(pwd):/backup alpine tar xzf /backup/caddy-data-backup.tar.gz -C /data
```

## 🆘 支持

如遇问题：
1. 查看日志: `docker-compose logs -f`
2. 检查 DNS 配置
3. 验证防火墙规则
4. 测试端口连通性: `telnet hechuqiu.net 443`
