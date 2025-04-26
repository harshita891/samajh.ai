class PresenceMonitor:
    def __init__(self, zone, max_lost_frames=30):
        self.zone = zone  # (x1, y1, x2, y2)
        self.objects_in_zone = {}
        self.max_lost_frames = max_lost_frames

    def update(self, active_objects):
        events = []

        for obj_id, bbox in active_objects.items():
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            in_zone = (self.zone[0] <= cx <= self.zone[2]) and (self.zone[1] <= cy <= self.zone[3])

            if obj_id not in self.objects_in_zone:
                self.objects_in_zone[obj_id] = {'in_zone': in_zone, 'lost': 0}
                if in_zone:
                    events.append(('entered', obj_id))
            else:
                last_status = self.objects_in_zone[obj_id]['in_zone']
                if in_zone and not last_status:
                    events.append(('entered', obj_id))
                elif not in_zone and last_status:
                    events.append(('exited', obj_id))
                self.objects_in_zone[obj_id]['in_zone'] = in_zone

        return events
