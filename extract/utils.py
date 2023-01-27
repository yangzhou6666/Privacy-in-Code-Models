import json
import re


def extract_uuid_from_string(s):
    uuids = re.findall('(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})', s)
    return uuids

def extract_email_from_string(s):
    emails = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", s)
    return emails

        