# SERVER.local.md example

这个文件是本地私有模板，不应提交真实密码。

可复制为：

`SERVER.local.md`

并填写仅保存在本机的敏感信息，例如：

```text
host: ssh.bj8.bz1.paratera.com
port: 2233
user: root@ackcs-00gjgqpy
password: <fill locally only>
remote_root: /root/fabric_run/fabric_defect_detection-main
```

正式同步时，这类文件应被 `.gitignore` 忽略。
