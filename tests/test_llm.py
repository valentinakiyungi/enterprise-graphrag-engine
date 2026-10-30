from app.llm import structured


class _Recorder:

  def __init__(self):
    self.calls = []

  def with_structured_output(self, schema, **kwargs):
    self.calls.append((schema, kwargs))
    return "wrapped"


def test_structured_default_method_is_not_forced():
  llm = _Recorder()
  assert structured(llm, dict) == "wrapped"
  assert llm.calls == [(dict, {})]


def test_structured_forwards_explicit_method():
  llm = _Recorder()
  structured(llm, dict, "function_calling")
  assert llm.calls == [(dict, {"method": "function_calling"})]
