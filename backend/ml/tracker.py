"""
Centroid-based Person Tracker for CrowdGuard.

This module provides person tracking across video frames using centroid matching.
It maintains unique identities for detected persons, computes velocities, and
manages track lifecycle (creation, update, removal).

**Validates: Requirements 3.1, 3.2**
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
from backend.ml.detector import Detection
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedPerson:
    """
    Represents a tracked person across frames.
    
    Attributes:
        track_id: Unique identifier for this track
        centroid: Current centroid position (x, y)
        velocity: Movement speed in pixels per frame
        bbox: Current bounding box (x1, y1, x2, y2)
        history: Rolling history of last 30 centroid positions
        frames_since_seen: Number of frames since last detection
    """
    track_id: int
    centroid: Tuple[int, int]
    velocity: float  # pixels per frame
    bbox: Tuple[int, int, int, int]
    history: List[Tuple[int, int]]  # Last 30 centroids
    frames_since_seen: int


class CentroidTracker:
    """
    Centroid-based object tracker for person tracking.
    
    Tracks persons across frames by matching centroids using Euclidean distance.
    Assigns unique IDs to new persons, maintains track history, computes velocities,
    and removes tracks that disappear for too long.
    
    **Validates: Requirements 3.1, 3.2**
    """
    
    def __init__(self, max_distance: int = 80, max_disappeared: int = 10):
        """
        Initialize centroid tracker.
        
        Args:
            max_distance: Maximum distance in pixels for centroid matching.
                         Detections further than this create new tracks. (default: 80)
            max_disappeared: Maximum number of consecutive frames a track can be
                           missing before removal. (default: 10)
                           
        **Validates: Requirements 3.3, 3.7**
        """
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        
        # Track storage: OrderedDict maintains insertion order
        # Maps track_id -> TrackedPerson
        self.objects = OrderedDict()
        
        # Tracks number of consecutive frames each track has been missing
        # Maps track_id -> frame_count
        self.disappeared = OrderedDict()
        
        # Counter for assigning unique track IDs
        self.next_object_id = 0
        
        logger.info(
            f"CentroidTracker initialized: max_distance={max_distance}, "
            f"max_disappeared={max_disappeared}"
        )
    
    def register(self, detection: Detection) -> int:
        """
        Register a new track with a unique ID.
        
        Creates a new TrackedPerson object and assigns it a unique track ID.
        Initializes the track with the detection's centroid, bbox, and zero velocity.
        
        Args:
            detection: Detection object to register as new track
            
        Returns:
            The assigned track ID
            
        **Validates: Requirements 3.1, 3.8**
        """
        track_id = self.next_object_id
        
        tracked_person = TrackedPerson(
            track_id=track_id,
            centroid=detection.centroid,
            velocity=0.0,  # Initial velocity is zero
            bbox=detection.bbox,
            history=[detection.centroid],  # Initialize history with first position
            frames_since_seen=0
        )
        
        self.objects[track_id] = tracked_person
        self.disappeared[track_id] = 0
        self.next_object_id += 1
        
        logger.debug(f"Registered new track: ID={track_id}, centroid={detection.centroid}")
        
        return track_id
    
    def deregister(self, track_id: int) -> None:
        """
        Remove a track from tracking.
        
        Deletes the track from both objects and disappeared dictionaries.
        
        Args:
            track_id: ID of track to remove
            
        **Validates: Requirement 3.7**
        """
        if track_id in self.objects:
            logger.debug(f"Deregistering track: ID={track_id}")
            del self.objects[track_id]
            del self.disappeared[track_id]
    
    def update(self, detections: List[Detection]) -> Dict[int, TrackedPerson]:
        """
        Update tracks with new detections from current frame.
        
        Matches detections to existing tracks using centroid distance,
        updates matched tracks, registers new tracks for unmatched detections,
        and removes tracks that have disappeared for too long.
        
        Args:
            detections: List of Detection objects from current frame
            
        Returns:
            Dictionary mapping track_id to TrackedPerson for all active tracks
            
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9**
        """
        # If no detections, mark all existing tracks as disappeared
        if len(detections) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                self.objects[track_id].frames_since_seen += 1
                
                # Remove tracks that have been missing too long
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister(track_id)
            
            return self.get_active_tracks()
        
        # If no existing tracks, register all detections as new tracks
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
            
            return self.get_active_tracks()
        
        # Match detections to existing tracks
        track_ids = list(self.objects.keys())
        track_centroids = np.array([self.objects[tid].centroid for tid in track_ids])
        detection_centroids = np.array([d.centroid for d in detections])
        
        # Compute distance matrix between all tracks and detections
        # Shape: (num_tracks, num_detections)
        distances = self._compute_distances(track_centroids, detection_centroids)
        
        # Find optimal matching using minimum distance
        matched_tracks, matched_detections = self._match_tracks(distances)
        
        # Update matched tracks
        for track_idx, detection_idx in zip(matched_tracks, matched_detections):
            track_id = track_ids[track_idx]
            detection = detections[detection_idx]
            
            self._update_track(track_id, detection)
        
        # Handle unmatched tracks (mark as disappeared)
        unmatched_tracks = set(range(len(track_ids))) - set(matched_tracks)
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.disappeared[track_id] += 1
            self.objects[track_id].frames_since_seen += 1
            
            # Remove if disappeared too long
            if self.disappeared[track_id] > self.max_disappeared:
                self.deregister(track_id)
        
        # Handle unmatched detections (register as new tracks)
        unmatched_detections = set(range(len(detections))) - set(matched_detections)
        for detection_idx in unmatched_detections:
            self.register(detections[detection_idx])
        
        return self.get_active_tracks()
    
    def _compute_distances(self, 
                          track_centroids: np.ndarray, 
                          detection_centroids: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between all tracks and detections.
        
        Args:
            track_centroids: Array of track centroids, shape (num_tracks, 2)
            detection_centroids: Array of detection centroids, shape (num_detections, 2)
            
        Returns:
            Distance matrix of shape (num_tracks, num_detections)
            
        **Validates: Requirement 3.3**
        """
        # Compute pairwise Euclidean distances
        # Broadcasting: (num_tracks, 1, 2) - (1, num_detections, 2)
        diff = track_centroids[:, np.newaxis, :] - detection_centroids[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        return distances
    
    def _match_tracks(self, distances: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Match tracks to detections using greedy minimum distance matching.
        
        Iteratively finds the minimum distance pair, matches them if distance
        is below max_distance threshold, and removes them from consideration.
        
        Args:
            distances: Distance matrix of shape (num_tracks, num_detections)
            
        Returns:
            Tuple of (matched_track_indices, matched_detection_indices)
            
        **Validates: Requirements 3.3, 3.4**
        """
        matched_tracks = []
        matched_detections = []
        
        # Create a copy to modify
        dist_copy = distances.copy()
        
        # Greedy matching: repeatedly find minimum distance
        while True:
            # Find minimum distance
            if dist_copy.size == 0:
                break
            
            min_idx = np.argmin(dist_copy)
            track_idx = min_idx // dist_copy.shape[1]
            detection_idx = min_idx % dist_copy.shape[1]
            min_distance = dist_copy[track_idx, detection_idx]
            
            # Check if distance is within threshold
            if min_distance > self.max_distance:
                break  # No more valid matches
            
            # Record match
            matched_tracks.append(track_idx)
            matched_detections.append(detection_idx)
            
            # Remove matched track and detection from consideration
            # Set their distances to infinity
            dist_copy[track_idx, :] = np.inf
            dist_copy[:, detection_idx] = np.inf
        
        return matched_tracks, matched_detections
    
    def _update_track(self, track_id: int, detection: Detection) -> None:
        """
        Update an existing track with a new detection.
        
        Updates centroid, bbox, computes velocity, maintains history,
        and resets disappeared counter.
        
        Args:
            track_id: ID of track to update
            detection: New detection to update track with
            
        **Validates: Requirements 3.2, 3.5, 3.6, 3.9**
        """
        tracked_person = self.objects[track_id]
        
        # Compute velocity (Euclidean distance from previous centroid)
        prev_centroid = tracked_person.centroid
        new_centroid = detection.centroid
        
        dx = new_centroid[0] - prev_centroid[0]
        dy = new_centroid[1] - prev_centroid[1]
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Update tracked person
        tracked_person.centroid = new_centroid
        tracked_person.bbox = detection.bbox
        tracked_person.velocity = float(velocity)
        tracked_person.frames_since_seen = 0
        
        # Update history (maintain last 30 positions)
        tracked_person.history.append(new_centroid)
        if len(tracked_person.history) > 30:
            tracked_person.history.pop(0)  # Remove oldest
        
        # Reset disappeared counter
        self.disappeared[track_id] = 0
        
        logger.debug(
            f"Updated track: ID={track_id}, centroid={new_centroid}, "
            f"velocity={velocity:.2f}"
        )
    
    def get_active_tracks(self) -> Dict[int, TrackedPerson]:
        """
        Get all currently active tracks.
        
        Returns:
            Dictionary mapping track_id to TrackedPerson for all active tracks
        """
        return dict(self.objects)
    
    def get_track_count(self) -> int:
        """
        Get the number of active tracks.
        
        Returns:
            Number of currently tracked persons
        """
        return len(self.objects)
    
    def reset(self) -> None:
        """
        Reset tracker state.
        
        Clears all tracks and resets the ID counter.
        Useful when starting a new session.
        """
        self.objects.clear()
        self.disappeared.clear()
        self.next_object_id = 0
        logger.info("Tracker reset: all tracks cleared")
