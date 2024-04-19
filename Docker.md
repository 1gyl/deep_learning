# # 什么是Docker

Docker是一个快捷、轻便的系统级虚拟化技术，开发者和系统管理员可以使用它构建具备所有必要依赖项的应用程序，并将其作为一个包发布。

Docker与其他工具虚拟化方式不同，每个虚拟机不需要单独的客户操作系统。

所有的Docker有效地共享一个主机系统内核。每个容器都在同一个操作系统中的格里用户空间中运行。

Docker镜像是一个描述容器应该如何表现的文件，而Docker容器是Docker镜像的运行（或停止）状态。

# Ubuntu安装Docker

## 添加Docker库

1、安装必要的证书并允许apt包管理器使用一下命令通过HTTPS使用存储库：

```
$ sudo apt install apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release
```

2、运行下列命令添加Docker的官方GPG密钥：

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

3.添加Docker官方库

```
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

```
sudo apt upate
```

## 安装Docker

```
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

安装完成后，可用以下命令验证Docker服务是否在运行

```
systemctl status docker
```

![image-20240417215522052](/home/gyl/.config/Typora/typora-user-images/image-20240417215522052.png)

## 测试 Docker

```
sudo docker run hello-world
```

![image-20240417215731692](/home/gyl/.config/Typora/typora-user-images/image-20240417215731692.png)

# 安装Docker Compose

Docker Compose是一个用于定义和运行多容器Docker应用程序的工具。

Docker Compose使得应用的部署和管理变得可移植，这意味着我们可以在不同的环境中轻松地重现相关应用结构。

Docker Compose可以与CI/CD（持续集成/持续部署）流程无缝集成，为自动化部署提供了便利。

我们可以使用 Pip 安装Docker Compose

```
pip install docker-compose
```

安装好后，可以使用以下命令检查版本

```
docker-compose --version
```

