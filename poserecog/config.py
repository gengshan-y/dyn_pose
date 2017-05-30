import ConfigParser

class lstm_config(object):
  config = ConfigParser.RawConfigParser()
  config.read('model/lstm.config')

  ctx = int(config.get('model', 'ctx'))
  batch_size = int(config.get('model', 'batch_size'))
  num_epochs = int(config.get('model', 'num_epochs'))
  disp_batches = int(config.get('model', 'disp_batches'))
  num_lstm_layer = int(config.get('model', 'num_lstm_layer'))
  input_dim = int(config.get('model', 'input_dim'))
  num_hidden = int(config.get('model', 'num_hidden'))
  num_embed = int(config.get('model', 'num_embed'))
  buckets = [int(config.get('model', 'buckets'))]
  dropout = float(config.get('model', 'dropout'))

  load_epoch = int(config.get('model', 'load_epoch'))
  test_buckets = int(config.get('model', 'test_buckets'))
