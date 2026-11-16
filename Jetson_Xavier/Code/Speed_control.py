class SpeedController:
    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.pwm = None
        self.started_at = None
        self.last_step_at = None
        self.in_startup = True
        self.has_speed_sample = False

    def update(self, now, speed, sampled_at):
        cfg = self.settings
        if self.pwm is None:
            self.pwm = cfg['start_pwm']
            self.started_at = now
            self.last_step_at = now
        fresh_speed = sampled_at > self.started_at and now - sampled_at < cfg['sensor_timeout_s']
        if fresh_speed:
            self.has_speed_sample = True
        elif self.has_speed_sample:
            raise TimeoutError('Потеря телеметрии скорости')
        elif now - self.started_at >= cfg['startup_timeout_s']:
            raise TimeoutError('Нет свежей скорости после старта')

        if self.in_startup:
            if self.pwm < cfg['ramp_pwm']:
                if now - self.last_step_at >= cfg['ramp_interval_s']:
                    self.pwm = min(cfg['ramp_pwm'], self.pwm + cfg['step_pwm'])
                    self.last_step_at = now
                return self.pwm
            if now - self.last_step_at < cfg['settle_time_s']:
                return self.pwm
            self.in_startup = False

        if not fresh_speed or sampled_at <= self.last_step_at:
            return self.pwm
        lower = cfg['target_kmh'] - cfg['deadband_kmh']
        upper = cfg['target_kmh'] + cfg['deadband_kmh']
        if lower <= speed <= upper:
            return self.pwm
        increasing = speed < lower
        interval = cfg['increase_interval_s'] if increasing else cfg['decrease_interval_s']
        if now - self.last_step_at >= interval:
            step = cfg['step_pwm'] if increasing else -cfg['step_pwm']
            self.pwm = max(cfg['min_pwm'], min(cfg['max_pwm'], self.pwm + step))
            self.last_step_at = now
        return self.pwm
