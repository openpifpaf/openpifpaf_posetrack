from collections import defaultdict
import logging

LOG = logging.getLogger(__name__)


class Signal:
    subscribers = defaultdict(list)

    @classmethod
    def emit(cls, name, *args, **kwargs):
        subscribers = cls.subscribers.get(name, [])
        LOG.debug('emit %s to %d subscribers', name, len(subscribers))
        for subscriber in subscribers:
            subscriber(*args, **kwargs)

    @classmethod
    def subscribe(cls, name, subscriber):
        LOG.debug('subscribe to %s', name)
        cls.subscribers[name].append(subscriber)
