class SpeedController:
    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.pwm = None
        self.started_at = None
        self.last_step_at = None

    def update(self, now, speed, sampled_at):
        cfg = self.settings
        if self.pwm is None:
            self.pwm = cfg['start_pwm']
            self.started_at = now
            self.last_step_at = now
        if sampled_at <= self.started_at:
            if now - self.started_at >= cfg['startup_timeout_s']:
                raise TimeoutError('Нет свежей скорости после старта')
            return self.pwm
        if now - sampled_at >= cfg['sensor_timeout_s']:
            raise TimeoutError('Потеря телеметрии скорости')
        error = cfg['target_kmh'] - speed
        if abs(error) <= cfg['deadband_kmh']:
            self.last_step_at = now
            return self.pwm
        interval = cfg['increase_interval_s'] if error > 0 else cfg['decrease_interval_s']
        if now - self.last_step_at >= interval:
            step = cfg['step_pwm'] if error > 0 else -cfg['step_pwm']
            self.pwm = max(cfg['min_pwm'], min(cfg['max_pwm'], self.pwm + step))
            self.last_step_at = now
        return self.pwm
