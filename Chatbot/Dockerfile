FROM centos:centos7.9.2009
RUN yum install epel-release -y && yum update -y && yum groupinstall "Development Tools" -y && yum install net-tools sudo vim wget python3-devel tar gzip gcc openssl-devel bzip2-devel libffi-devel zlib-devel make expect -y && wget -P /opt https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz && cd /opt && tar xzf Python-3.9.6.tgz && cd /opt/Python-3.9.6  && sudo ./configure --enable-optimizations && sudo make altinstall 
RUN yum install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-7-x86_64/pgdg-redhat-repo-latest.noarch.rpm && yum install -y postgresql10-server
COPY requirements.txt /
ADD startup.sh /
RUN chmod a+x /startup.sh
COPY . /var/www/production/Chatbot
# ENTRYPOINT ["/startup.sh"]
CMD ["/usr/sbin/init"]

