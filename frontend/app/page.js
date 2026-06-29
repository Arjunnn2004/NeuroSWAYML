"use client";

import React, { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { Play, Square, Activity, RefreshCcw, Camera, AlertTriangle, FileText, Database, ShieldAlert, Trash2, Star } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:5000';
const STATUS_FIELDS = [
  'running',
  'offline',
  'fps',
  'domain',
  'risk_class',
  'fall_detected',
  'sway_idx',
  'threshold_issues',
  'balance_tests'
];

const hasStatusChanged = (prev, next) => (
  STATUS_FIELDS.some((field) => JSON.stringify(prev[field]) !== JSON.stringify(next[field]))
);

const formatImageDate = (filename) => {
  const match = filename.match(/fall_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  if (!match) return { date: filename.replace('.jpg', ''), time: '' };

  const [, y, m, d, h, min, s] = match;
  const dateObj = new Date(y, m - 1, d, h, min, s);
  return {
    date: dateObj.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }),
    time: dateObj.toLocaleTimeString()
  };
};

const DetectionCard = memo(function DetectionCard({ img, isFav, onToggleFavorite, onDeleteImage }) {
  const { date, time } = useMemo(() => formatImageDate(img), [img]);

  return (
    <article className="glass-card image-card">
      <div className="image-thumb-wrapper">
        <img src={`${API_BASE}/api/images/${img}`} className="image-thumb" alt={img} loading="lazy" decoding="async" />
        <div className="image-overlay-controls">
          <button
            type="button"
            className={`img-btn fav ${isFav ? 'active' : ''}`}
            onClick={() => onToggleFavorite(img)}
            title="Toggle Favorite"
            aria-label="Toggle favorite"
          >
            <Star size={16} fill={isFav ? "currentColor" : "none"} />
          </button>
          <button
            type="button"
            className="img-btn delete"
            onClick={() => onDeleteImage(img)}
            title="Delete"
            aria-label="Delete detection"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>
      <div className="image-info">
        <div className="image-date">{date}</div>
        {time && <div className="image-time-sub">{time}</div>}
      </div>
    </article>
  );
});

export default function Dashboard() {
  const [status, setStatus] = useState({ running: false });
  const [logs, setLogs] = useState([]);
  const [images, setImages] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [favorites, setFavorites] = useState({});

  useEffect(() => {
    const savedFavs = localStorage.getItem('neurosway_favs');
    if (savedFavs) {
      try {
        setFavorites(JSON.parse(savedFavs));
      } catch {
        localStorage.removeItem('neurosway_favs');
      }
    }
  }, []);

  const toggleFavorite = useCallback((imgName) => {
    setFavorites(prev => {
      const next = { ...prev, [imgName]: !prev[imgName] };
      localStorage.setItem('neurosway_favs', JSON.stringify(next));
      return next;
    });
  }, []);

  const deleteImage = useCallback(async (imgName) => {
    if (!confirm(`Are you sure you want to delete ${imgName}?`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/images/${imgName}`, { method: 'DELETE' });
      if (res.ok) {
        setImages(prev => prev.filter(img => img !== imgName));
      }
    } catch (e) {
      console.error("Delete failed", e);
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(prev => hasStatusChanged(prev, data) ? data : prev);
      }
    } catch (e) {
      setStatus(prev => prev.offline ? prev : { running: false, offline: true });
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const [logsRes, imgRes] = await Promise.all([
        fetch(`${API_BASE}/api/logs`),
        fetch(`${API_BASE}/api/images`)
      ]);
      const logsData = await logsRes.json();
      const imgData = await imgRes.json();
      setLogs(logsData.reverse());
      setImages(imgData);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, [fetchHistory, fetchStatus]);

  const handleStart = async () => {
    await fetch(`${API_BASE}/api/start`);
    fetchStatus();
  };

  const handleStop = async () => {
    await fetch(`${API_BASE}/api/stop`);
    fetchStatus();
  };

  const handleCalibrate = async () => {
    await fetch(`${API_BASE}/api/calibrate`);
  };

  const isRunning = status.running;
  const isOffline = status.offline;

  return (
    <div className="container">
      <header className="dashboard-header">
        <div className="header-title">
          <Activity color="var(--accent)" size={32} />
          NeuroSWAY Dashboard
          <span className="header-badge">ML Core v1.0</span>
        </div>
        
        <div className="status-pill glass-panel">
          <div className={`status-dot ${isOffline ? 'dot-offline' : (isRunning ? 'dot-online' : 'dot-offline')}`} />
          {isOffline ? 'Engine Offline' : (isRunning ? 'Engine Active' : 'Engine Idle')}
        </div>
      </header>

      <div className="dashboard-grid">
        <aside className="controls-sidebar">
          <div className="glass-card control-group">
            <h3 className="control-title"><ShieldAlert size={20} /> Controls</h3>
            
            <button 
              className={`btn ${isRunning ? 'btn-danger' : 'btn-primary'}`}
              onClick={isRunning ? handleStop : handleStart}
              disabled={isOffline}
            >
              {isRunning ? <><Square size={18} /> Stop Analysis</> : <><Play size={18} /> Start Engine</>}
            </button>

            <button 
              className="btn btn-accent" 
              onClick={handleCalibrate}
              disabled={!isRunning}
            >
              <RefreshCcw size={18} /> Recalibrate Base
            </button>
          </div>

          {isRunning && (
              <div className="glass-card control-group">
                <h3 className="control-title"><Database size={20} /> Live Telemetry</h3>
                
                <div className="stat-row">
                  <span className="stat-label">FPS</span>
                  <span className="stat-val">{status.fps?.toFixed(1) || 0}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Domain</span>
                  <span className="stat-val text-capitalize">{status.domain || '-'}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Risk Class</span>
                  <span className={`stat-val ${status.risk_class > 1 ? 'high-risk' : ''}`}>
                    Level {status.risk_class || 0}
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Gait Sway</span>
                  <span className={`stat-val ${status.sway_idx > 2.5 ? 'warning' : ''}`}>
                    {status.sway_idx?.toFixed(2) || 0.0}
                  </span>
                </div>
                
                {status.threshold_issues?.length > 0 && (
                  <div className="alert-list">
                    <div className="alert-list-title">Active Alerts</div>
                    {status.threshold_issues.map((issue, i) => (
                      <div key={i} className="alert-item">{issue}</div>
                    ))}
                  </div>
                )}
              </div>
            )}

          {isRunning && status.balance_tests && (
              <div className="glass-card control-group">
                <h3 className="control-title"><Activity size={20} /> Balance Tests</h3>

                <div className="stat-row">
                  <span className="stat-label">Current Stance</span>
                  <span className={`stat-val ${status.balance_tests.stable ? '' : 'warning'}`}>
                    {status.balance_tests.stance_label || '-'}
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Four-Stage</span>
                  <span className="stat-val">
                    {status.balance_tests.four_stage?.passed_count || 0}/{status.balance_tests.four_stage?.total_count || 4}
                  </span>
                </div>

                <div className="balance-stage-list">
                  {status.balance_tests.four_stage?.stages?.map((stage) => (
                    <div key={stage.key} className={`balance-stage ${stage.active ? 'active' : ''} ${stage.passed ? 'passed' : ''}`}>
                      <span>{stage.name}</span>
                      <strong>
                        {stage.active ? (stage.hold_seconds?.toFixed?.(1) || stage.hold_seconds || 0) : (stage.best_seconds?.toFixed?.(1) || stage.best_seconds || 0)}
                        /{stage.target_seconds}s
                      </strong>
                    </div>
                  ))}
                </div>

                <div className="stat-row">
                  <span className="stat-label">Flamingo Hold</span>
                  <span className={`stat-val ${status.balance_tests.flamingo?.passed ? '' : 'warning'}`}>
                    {status.balance_tests.flamingo?.hold_seconds?.toFixed?.(1) || 0}/{status.balance_tests.flamingo?.target_seconds || 60}s
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Flamingo Best</span>
                  <span className="stat-val">
                    {status.balance_tests.flamingo?.best_seconds?.toFixed?.(1) || 0}s
                  </span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Flamingo Losses</span>
                  <span className={`stat-val ${(status.balance_tests.flamingo?.losses || 0) > 3 ? 'high-risk' : ''}`}>
                    {status.balance_tests.flamingo?.losses || 0}
                  </span>
                </div>
              </div>
            )}

          <div className="glass-card control-group">
            <h3 className="control-title"><FileText size={20} /> Session Overview</h3>
            <div className="stat-row">
              <span className="stat-label">Archived Sessions</span>
              <span className="stat-val">{logs.length}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Detected Falls</span>
              <span className="stat-val">{images.length}</span>
            </div>
          </div>
        </aside>

        <section>
          <div className="video-container">
            {isRunning ? (
              <img 
                src={`${API_BASE}/video_feed`} 
                className="video-feed" 
                alt="Live ML Feed" 
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            ) : (
              <div className="video-placeholder">
                <Camera size={48} opacity={0.5} />
                <p>{isOffline ? 'API Server Unreachable' : 'Engine is stopped. Press Start to begin stream.'}</p>
              </div>
            )}
            
            {status.fall_detected && (
                <div className="fall-alert">
                  <AlertTriangle />
                  FALL DETECTED
                </div>
              )}
          </div>

          <div className="images-section">
            <h2 className="section-title"><Camera size={24} /> Recent Detections</h2>
            <div className="image-grid">
              {images.map((img) => (
                <DetectionCard
                  key={img}
                  img={img}
                  isFav={Boolean(favorites[img])}
                  onToggleFavorite={toggleFavorite}
                  onDeleteImage={deleteImage}
                />
              ))}
              {images.length === 0 && !isLoading && (
                <p className="empty-state">No detections found.</p>
              )}
            </div>
          </div>
          
        </section>
      </div>
    </div>
  );
}
