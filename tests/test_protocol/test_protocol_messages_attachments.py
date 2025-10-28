from gpt2giga.config import ProxyConfig
from gpt2giga.protocol import RequestTransformer


class DummyAttachmentProc:
    def __init__(self):
        self.calls = 0

    def upload_image(self, url):
        self.calls += 1
        return f"file_{self.calls}"


def test_transform_messages_with_images_and_limit_two_per_message():
    cfg = ProxyConfig()
    ap = DummyAttachmentProc()
    rt = RequestTransformer(cfg, attachment_processor=ap)

    content = [
        {"type": "text", "text": "t1"},
        {"type": "image_url", "image_url": {"url": "u1"}},
        {"type": "image_url", "image_url": {"url": "u2"}},
        {"type": "image_url", "image_url": {"url": "u3"}},
    ]
    messages = [{"role": "user", "content": content}]
    out = rt.transform_messages(messages)

    assert out[0]["attachments"] == ["file_1", "file_2"]


def test_transform_messages_total_attachments_limit_ten():
    cfg = ProxyConfig()
    ap = DummyAttachmentProc()
    rt = RequestTransformer(cfg, attachment_processor=ap)

    many = [{"type": "image_url", "image_url": {"url": f"u{i}"}} for i in range(20)]
    messages = [
        {"role": "user", "content": many[:5]},
        {"role": "user", "content": many[5:15]},
    ]
    out = rt.transform_messages(messages)
    total = sum(len(m.get("attachments", [])) for m in out)
    assert total == 4
