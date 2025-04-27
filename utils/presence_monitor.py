class PresenceMonitor:
    def __init__(self, zone, max_lost_frames, buffer_zone=10):
        self.zone = zone
        self.max_lost_frames = max_lost_frames
        self.object_lost_frames = {}
        self.in_zone_status = {}
        self.buffer_zone = buffer_zone
        # Validate zone coordinates
        if not (0 <= zone[0] < zone[2] and 0 <= zone[1] < zone[3]):
            raise ValueError(f"Invalid zone coordinates: {zone}")

    def update(self, active_objects):
        events = []
        current_in_zone = set()

        for obj_id, bbox in active_objects.items():
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            in_zone = (
                self.zone[0] - self.buffer_zone <= cx <= self.zone[2] + self.buffer_zone
                and self.zone[1] - self.buffer_zone <= cy <= self.zone[3] + self.buffer_zone
            )

            # Debugging output for tracking object behavior
            print(f"Object {obj_id}: Center (cx: {cx:.1f}, cy: {cy:.1f}), In Zone: {in_zone}, Zone: {self.zone}")

            if in_zone:
                # Object is in zone
                current_in_zone.add(obj_id)
                if obj_id not in self.in_zone_status or not self.in_zone_status[obj_id]:
                    events.append(('new', obj_id))  # Trigger new event when an object enters the zone
                    print(f"New event triggered for Object {obj_id}")
                    self.in_zone_status[obj_id] = True
                self.object_lost_frames[obj_id] = 0  # Reset lost frames count when the object is inside
            else:
                # Object is not in zone
                if obj_id not in self.object_lost_frames:
                    self.object_lost_frames[obj_id] = 1  # Initialize lost frames count if it's a new object
                else:
                    self.object_lost_frames[obj_id] += 1  # Increment lost frames count if the object stays out

        # Check for missing objects (objects outside the zone for too long)
        for obj_id in list(self.in_zone_status.keys()):
            if obj_id not in current_in_zone and self.in_zone_status.get(obj_id, False):
                self.object_lost_frames[obj_id] = self.object_lost_frames.get(obj_id, 0) + 1
                if self.object_lost_frames[obj_id] > self.max_lost_frames:
                    events.append(('missing', obj_id))  # Trigger missing event when object is out too long
                    print(f"Missing event triggered for Object {obj_id}")
                    self.in_zone_status[obj_id] = False  # Mark as missing

        return events
