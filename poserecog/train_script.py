import mxnet as mx
import time

def _as_list(obj):
    """A utility function that treat the argument as a list.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list, return it. Otherwise, return `[obj]` as a single-element list.
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]

def fit(model, train_data, eval_data=None, eval_metric='acc',
        epoch_end_callback=None, batch_end_callback=None, kvstore='local',
        optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
        eval_end_callback=None,
        eval_batch_end_callback=None, initializer=mx.initializer.Uniform(0.01),
        arg_params=None, aux_params=None, allow_missing=False,
        force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
        validation_metric=None, monitor=None,val_callback=None):


    assert num_epoch is not None, 'please specify number of epochs'

    model.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
              for_training=True, force_rebind=force_rebind)
    if monitor is not None:
        model.install_monitor(monitor)
    model.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                     allow_missing=allow_missing, force_init=force_init)
    model.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                        optimizer_params=optimizer_params)

    if validation_metric is None:
        validation_metric = eval_metric
    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)

    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(begin_epoch, num_epoch):
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(data_iter)
        while not end_of_batch:
            data_batch = next_data_batch
            if monitor is not None:
                monitor.tic()
            model.forward_backward(data_batch)
            model.update()
            try:
                # pre fetch next batch
                next_data_batch = next(data_iter)
                model.prepare(next_data_batch)
            except StopIteration:
                end_of_batch = True

            model.update_metric(eval_metric, data_batch.label)

            if monitor is not None:
                monitor.toc_print()

            if batch_end_callback is not None:
                batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)
            nbatch += 1

        # one epoch of training is finished
        for name, val in eval_metric.get_name_value():
            model.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
        toc = time.time()
        model.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

        # sync aux params across devices
        arg_params, aux_params = model.get_params()
        model.set_params(arg_params, aux_params)

        if epoch_end_callback is not None:
            for callback in _as_list(epoch_end_callback):
                callback(epoch, model.symbol, arg_params, aux_params)

        #----------------------------------------
        # evaluation on validation set
        if eval_data:
            res = model.score(eval_data, validation_metric,
                             score_end_callback=eval_end_callback,
                             batch_end_callback=eval_batch_end_callback, epoch=epoch)
            #TODO: pull this into default
            for name, val in res:
                model.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            if val_callback is not None:
              val_callback(res)

        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()
