import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.header import Header

def send_email(sender, password, receiver, smtp_server, smtp_port, email_message, subject, attachment_path=None):
    # Verifica se os parâmetros de autenticação são válidos
    if not sender or not password:
        raise ValueError("O remetente e a senha devem ser fornecidos e não podem ser None.")

    msg = MIMEMultipart()
    msg['To'] = Header(receiver)
    msg['From'] = Header(sender)
    msg['Subject'] = Header(subject)

    # Adiciona o corpo do email
    msg.attach(MIMEText(email_message, 'plain', 'utf-8'))

    # Adiciona o anexo, se houver
    if attachment_path:
        with open(attachment_path, 'rb') as attachment:
            att = MIMEApplication(attachment.read(), _subtype='txt')
            att.add_header('Content-Disposition', 'attachment', filename=attachment_path)
            msg.attach(att)

    # Configuração do servidor SMTP
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.ehlo()

        if not sender:
            raise ValueError("Sender email is None")
        if not password:
            raise ValueError("Password is None")
        
        server.login(sender, password)
        text = msg.as_string()
        server.sendmail(sender, receiver, text)
    except smtplib.SMTPAuthenticationError as e:
        raise ValueError("Falha na autenticação: verifique o endereço de e-mail e a senha.") from e
    except Exception as e:
        raise ValueError("Ocorreu um erro ao enviar o e-mail.") from e
    finally:
        server.quit()