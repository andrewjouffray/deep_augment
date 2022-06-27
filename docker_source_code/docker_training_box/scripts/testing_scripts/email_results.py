# Import smtplib library to send email in python.
import smtplib
# Import MIMEText, MIMEImage and MIMEMultipart module.
from email.MIMEImage import MIMEImage
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
import os

# Define the source and target email address.
strFrom = os.getenv('G_EMAIL')
strTo = os.getenv('G_EMAIL')

# Create an instance of MIMEMultipart object, pass 'related' as the constructor parameter.
msgRoot = MIMEMultipart('related')
# Set the email subject.
msgRoot['Subject'] = 'Latest Testing results.'
# Set the email from email address.
msgRoot['From'] = strFrom
# Set the email to email address.
msgRoot['To'] = strTo

# Set the multipart email preamble attribute value. Please refer https://docs.python.org/3/library/email.message.html to learn more.
msgRoot.preamble = '====================================================='

# Create a 'alternative' MIMEMultipart object. We will use this object to save plain text format content.
msgAlternative = MIMEMultipart('alternative')
# Attach the bove object to the root email message.
msgRoot.attach(msgAlternative)

# Create a MIMEText object, this object contains the plain text content.
msgText = MIMEText('Here is a graph of the testing results:.')
# Attach the MIMEText object to the msgAlternative object.
msgAlternative.attach(msgText)

# Open a file object to read the image file, the image file is located in the file path it provide.
fp = open('/workdir/testing/results.png', 'rb')
# Create a MIMEImage object with the above file object.
msgImage = MIMEImage(fp.read())
# Do not forget close the file object after using it.
fp.close()

# Add 'Content-ID' header value to the above MIMEImage object to make it refer to the image source (src="cid:image1") in the Html content.
msgImage.add_header('Content-ID', '<image1>')
# Attach the MIMEImage object to the email body.
msgRoot.attach(msgImage)

# Connect to the SMTP server.


server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
server.ehlo()
server.login(os.getenv('G_EMAIL'), os.getenv('G_PASS'))

# Send email with the smtp object sendmail method.
smtp.sendmail(strFrom, strTo, msgRoot.as_string())
# Quit the SMTP server after sending the email.
smtp.quit()
