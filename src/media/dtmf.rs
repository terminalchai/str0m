//! Telephone events (DTMF), RFC 4733.
//!
//! Telephone events carry named signals such as DTMF digits over RTP using a
//! dedicated `telephone-event` payload type, negotiated inside an audio m-line.
//! Each event is reported by a small, fixed-size RTP payload (4 bytes) that is
//! resent across several packets: the packets of one event share an RTP
//! timestamp, carry a growing duration, and the final packet (with the end bit
//! set) is repeated for robustness (RFC 4733 §2.5).
//!
//! Sending is done with [`Writer::write_dtmf`][crate::media::Writer::write_dtmf]
//! and received tones are surfaced as [`Event::DtmfEvent`][crate::Event::DtmfEvent].

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::rtp_::{ExtensionValues, MediaTime, Mid, Pt};

use super::ToPayload;

/// Number of times the final packet of an event (with the end bit set) is sent.
///
/// RFC 4733 §2.5.1.4 recommends resending the final packet for robustness.
const END_PACKET_REPEATS: u8 = 3;

/// Default per-packet interval for a DTMF tone (its ptime).
const DEFAULT_PACKET_INTERVAL: Duration = Duration::from_millis(20);

/// Default volume for a DTMF tone, in -dBm0 (RFC 4733 §2.5.2.1).
const DEFAULT_VOLUME: u8 = 10;

/// A DTMF digit or telephony event (RFC 4733 §3.2, RFC 4734).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Dtmf {
    D0,
    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    D7,
    D8,
    D9,
    /// The `*` key.
    Star,
    /// The `#` key.
    Pound,
    A,
    B,
    C,
    D,
    /// Hook flash.
    Flash,
}

impl Dtmf {
    /// The RFC 4733 event code for this event.
    pub fn event_code(&self) -> u8 {
        use Dtmf::*;
        match self {
            D0 => 0,
            D1 => 1,
            D2 => 2,
            D3 => 3,
            D4 => 4,
            D5 => 5,
            D6 => 6,
            D7 => 7,
            D8 => 8,
            D9 => 9,
            Star => 10,
            Pound => 11,
            A => 12,
            B => 13,
            C => 14,
            D => 15,
            Flash => 16,
        }
    }

    /// Creates a [`Dtmf`] from an RFC 4733 event code, if it is a known event.
    pub fn from_event_code(code: u8) -> Option<Dtmf> {
        use Dtmf::*;
        Some(match code {
            0 => D0,
            1 => D1,
            2 => D2,
            3 => D3,
            4 => D4,
            5 => D5,
            6 => D6,
            7 => D7,
            8 => D8,
            9 => D9,
            10 => Star,
            11 => Pound,
            12 => A,
            13 => B,
            14 => C,
            15 => D,
            16 => Flash,
            _ => return None,
        })
    }

    /// The dialpad character for this event, if any.
    pub fn to_char(&self) -> Option<char> {
        use Dtmf::*;
        Some(match self {
            D0 => '0',
            D1 => '1',
            D2 => '2',
            D3 => '3',
            D4 => '4',
            D5 => '5',
            D6 => '6',
            D7 => '7',
            D8 => '8',
            D9 => '9',
            Star => '*',
            Pound => '#',
            A => 'A',
            B => 'B',
            C => 'C',
            D => 'D',
            Flash => return None,
        })
    }

    /// Parses a dialpad character (`0`–`9`, `*`, `#`, `A`–`D`) into an event.
    pub fn from_char(c: char) -> Option<Dtmf> {
        use Dtmf::*;
        Some(match c.to_ascii_uppercase() {
            '0' => D0,
            '1' => D1,
            '2' => D2,
            '3' => D3,
            '4' => D4,
            '5' => D5,
            '6' => D6,
            '7' => D7,
            '8' => D8,
            '9' => D9,
            '*' => Star,
            '#' => Pound,
            'A' => A,
            'B' => B,
            'C' => C,
            'D' => D,
            _ => return None,
        })
    }
}

/// A single telephone-event (RFC 4733) RTP payload.
///
/// This is the 4-byte payload carried by one `telephone-event` RTP packet. Use
/// it to build or inspect raw telephone-event packets; for sending DTMF tones at
/// a higher level, use [`Writer::write_dtmf`][crate::media::Writer::write_dtmf].
///
/// ```text
///  0                   1                   2                   3
///  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// |     event     |E|R| volume    |          duration             |
/// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TelephoneEventPayload {
    /// The event code (RFC 4733 §2.5.1.1). See [`Dtmf::from_event_code`].
    pub event: u8,

    /// The end bit: set on the final packet(s) of an event (RFC 4733 §2.5.1.3).
    pub end: bool,

    /// The volume of the event, in -dBm0 (0–63, RFC 4733 §2.5.1.4).
    ///
    /// Only meaningful for DTMF events. Higher values are quieter.
    pub volume: u8,

    /// The duration of the event so far, in RTP timestamp units (samples at the
    /// payload clock rate).
    pub duration: u16,
}

impl TelephoneEventPayload {
    /// Parses a telephone-event payload from the 4 payload bytes of an RTP packet.
    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < 4 {
            return None;
        }
        Some(TelephoneEventPayload {
            event: buf[0],
            end: buf[1] & 0x80 != 0,
            volume: buf[1] & 0x3f,
            duration: u16::from_be_bytes([buf[2], buf[3]]),
        })
    }

    /// Serializes this payload to its 4 wire bytes.
    pub fn to_bytes(&self) -> [u8; 4] {
        let [d0, d1] = self.duration.to_be_bytes();
        let end = if self.end { 0x80 } else { 0x00 };
        [self.event, end | (self.volume & 0x3f), d0, d1]
    }
}

/// A received telephone event (DTMF), surfaced by
/// [`Event::DtmfEvent`][crate::Event::DtmfEvent].
///
/// One `DtmfEvent` is emitted per completed tone, once its final (end) packet
/// has been received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtmfEvent {
    /// The media (m-line) the event was received on.
    pub mid: Mid,

    /// The raw RFC 4733 event code.
    pub event: u8,

    /// The decoded DTMF digit, if the event code is a known DTMF event.
    pub dtmf: Option<Dtmf>,

    /// The reported volume, in -dBm0 (0–63).
    pub volume: u8,

    /// The total duration of the tone, in the payload clock rate.
    pub duration: MediaTime,
}

/// A tone queued for sending by [`DtmfSender`].
#[derive(Debug)]
struct QueuedTone {
    pt: Pt,
    event: u8,
    volume: u8,
    /// The full duration of the tone in samples at `clock_rate`.
    total_samples: u32,
    /// The number of samples each packet advances the duration.
    step_samples: u32,
    /// RTP timestamp shared by every packet of this tone.
    rtp_time: MediaTime,
    /// Intended send time of the first packet.
    start: Instant,
    /// The per-packet interval.
    interval: Duration,
}

/// The tone currently being transmitted by [`DtmfSender`].
#[derive(Debug)]
struct ActiveTone {
    tone: QueuedTone,
    /// Total number of packets emitted so far (for the marker on the first).
    packets: u32,
    /// Current event duration in samples (grows while playing, capped at total).
    duration: u32,
    /// Whether the tone has reached its full duration.
    ended: bool,
    /// Number of end packets (end bit set) emitted so far.
    end_sent: u8,
    /// When the next packet is due.
    next_at: Instant,
}

/// Generates the RTP packet series for outgoing DTMF tones (RFC 4733).
///
/// Tones are played back-to-back in the order they are queued. Each tone is
/// emitted as a series of packets that share an RTP timestamp and carry a
/// growing duration, with the marker bit on the first packet and the final
/// packet repeated with the end bit set.
#[derive(Debug, Default)]
pub(crate) struct DtmfSender {
    queue: VecDeque<QueuedTone>,
    active: Option<ActiveTone>,
}

impl DtmfSender {
    /// Queues a DTMF tone for sending.
    pub fn push(
        &mut self,
        pt: Pt,
        rtp_time: MediaTime,
        wallclock: Instant,
        event: u8,
        volume: u8,
        duration: Duration,
        clock_rate: crate::rtp_::Frequency,
    ) {
        let hz = clock_rate.get() as u64;

        let total_samples = ((duration.as_micros() as u64 * hz) / 1_000_000).max(1) as u32;

        let interval = DEFAULT_PACKET_INTERVAL;
        let step_samples = ((interval.as_micros() as u64 * hz) / 1_000_000).max(1) as u32;

        self.queue.push_back(QueuedTone {
            pt,
            event,
            volume: volume.min(0x3f),
            total_samples,
            step_samples,
            rtp_time,
            start: wallclock,
            interval,
        });
    }

    /// The next time [`DtmfSender::poll`] should be called, if any packet is
    /// pending.
    pub fn poll_timeout(&self) -> Option<Instant> {
        if let Some(active) = &self.active {
            return Some(active.next_at);
        }
        // A queued but not-yet-active tone becomes active at its start time.
        self.queue.front().map(|t| t.start)
    }

    /// Produces the next due telephone-event payload, if one is ready at `now`.
    ///
    /// Returns `None` when nothing is due yet or there are no pending tones.
    pub fn poll(&mut self, now: Instant) -> Option<ToPayload> {
        // Activate the next queued tone if nothing is playing.
        if self.active.is_none() {
            let tone = self.queue.pop_front()?;
            let next_at = tone.start;
            self.active = Some(ActiveTone {
                tone,
                packets: 0,
                duration: 0,
                ended: false,
                end_sent: 0,
                next_at,
            });
        }

        // unwrap: set just above if it was None.
        let active = self.active.as_mut().unwrap();

        if now < active.next_at {
            return None;
        }

        let first = active.packets == 0;

        // Advance the event state. The packet at which the duration first reaches
        // the total is the first end packet; it is then repeated for robustness
        // (RFC 4733 §2.5.1.4).
        let (end, last) = if !active.ended {
            active.duration = active
                .duration
                .saturating_add(active.tone.step_samples)
                .min(active.tone.total_samples);

            if active.duration >= active.tone.total_samples {
                active.ended = true;
                active.end_sent = 1;
                (true, END_PACKET_REPEATS <= 1)
            } else {
                (false, false)
            }
        } else {
            active.end_sent = active.end_sent.saturating_add(1);
            (true, active.end_sent >= END_PACKET_REPEATS)
        };

        let payload = TelephoneEventPayload {
            event: active.tone.event,
            end,
            volume: active.tone.volume,
            duration: active.duration.min(u16::MAX as u32) as u16,
        };

        let data: Arc<[u8]> = Arc::from(payload.to_bytes().as_slice());

        let to_payload = ToPayload {
            pt: active.tone.pt,
            rid: None,
            wallclock: active.next_at,
            rtp_time: active.tone.rtp_time,
            start_of_talk_spurt: first,
            data,
            ext_vals: ExtensionValues::default(),
        };

        // Advance schedule.
        active.packets = active.packets.saturating_add(1);
        active.next_at += active.tone.interval;

        if last {
            self.active = None;
        }

        Some(to_payload)
    }
}

/// Reassembles incoming telephone-event (RFC 4733) packets into completed
/// [`DtmfEvent`]s.
#[derive(Debug, Default)]
pub(crate) struct DtmfReceiver {
    current: Option<InProgress>,
    ready: VecDeque<DtmfEvent>,
}

/// State for the telephone event currently being received.
#[derive(Debug)]
struct InProgress {
    /// The RTP timestamp identifying this event.
    timestamp: u64,
    event: u8,
    volume: u8,
    /// Largest duration seen across the event's packets.
    duration: u16,
    /// Whether a completed `DtmfEvent` has already been emitted for this event.
    emitted: bool,
}

impl DtmfReceiver {
    /// Feeds one depacketized telephone-event payload into the aggregator.
    ///
    /// `time` is the RTP timestamp of the packet (which identifies the event).
    pub fn feed(&mut self, mid: Mid, time: MediaTime, payload: TelephoneEventPayload) {
        let ts = time.numer();

        let is_same = self
            .current
            .as_ref()
            .map(|c| c.timestamp == ts && c.event == payload.event)
            .unwrap_or(false);

        if !is_same {
            // A new event started. If the previous one never emitted (its end
            // packets were lost), emit it now as a best effort.
            self.flush_current(mid, time.frequency());

            self.current = Some(InProgress {
                timestamp: ts,
                event: payload.event,
                volume: payload.volume,
                duration: payload.duration,
                emitted: false,
            });
        }

        // unwrap: set just above if it wasn't the same event.
        let current = self.current.as_mut().unwrap();
        current.duration = current.duration.max(payload.duration);
        current.volume = payload.volume;

        if payload.end && !current.emitted {
            current.emitted = true;
            let ev = current.to_event(mid, time.frequency());
            self.ready.push_back(ev);
        }
    }

    /// Pops the next completed [`DtmfEvent`], if any.
    pub fn poll(&mut self) -> Option<DtmfEvent> {
        self.ready.pop_front()
    }

    fn flush_current(&mut self, mid: Mid, freq: crate::rtp_::Frequency) {
        if let Some(current) = &self.current {
            if !current.emitted {
                let ev = current.to_event(mid, freq);
                self.ready.push_back(ev);
            }
        }
        self.current = None;
    }
}

impl InProgress {
    fn to_event(&self, mid: Mid, freq: crate::rtp_::Frequency) -> DtmfEvent {
        DtmfEvent {
            mid,
            event: self.event,
            dtmf: Dtmf::from_event_code(self.event),
            volume: self.volume,
            duration: MediaTime::new(self.duration as u64, freq),
        }
    }
}

/// Default volume used by the high-level DTMF sending API.
pub(crate) const fn default_volume() -> u8 {
    DEFAULT_VOLUME
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::rtp_::Frequency;

    #[test]
    fn payload_roundtrip() {
        let p = TelephoneEventPayload {
            event: 5,
            end: true,
            volume: 10,
            duration: 1600,
        };
        let bytes = p.to_bytes();
        assert_eq!(bytes, [0x05, 0x8a, 0x06, 0x40]);
        assert_eq!(TelephoneEventPayload::parse(&bytes), Some(p));
    }

    #[test]
    fn dtmf_char_roundtrip() {
        for c in "0123456789*#ABCD".chars() {
            let d = Dtmf::from_char(c).unwrap();
            assert_eq!(d.to_char(), Some(c));
            assert_eq!(Dtmf::from_event_code(d.event_code()), Some(d));
        }
    }

    #[test]
    fn sender_emits_series_with_marker_and_end_repeats() {
        let mut s = DtmfSender::default();
        let start = Instant::now();
        // 100 ms tone at 8 kHz, 20 ms packets => 5 playing packets + repeats.
        s.push(
            101.into(),
            MediaTime::new(0, Frequency::EIGHT_KHZ),
            start,
            5,
            10,
            Duration::from_millis(100),
            Frequency::EIGHT_KHZ,
        );

        let mut now = start;
        let mut payloads = vec![];
        for _ in 0..20 {
            while let Some(tp) = s.poll(now) {
                payloads.push(tp);
            }
            now += Duration::from_millis(20);
        }

        // First packet has the marker (start of talkspurt).
        assert!(payloads[0].start_of_talk_spurt);
        assert!(!payloads[1].start_of_talk_spurt);

        // All packets share the same RTP timestamp.
        for p in &payloads {
            assert_eq!(p.rtp_time.numer(), 0);
        }

        // The final three packets carry the end bit and full duration.
        let parsed: Vec<_> = payloads
            .iter()
            .map(|p| TelephoneEventPayload::parse(&p.data).unwrap())
            .collect();
        let end_count = parsed.iter().filter(|p| p.end).count();
        assert_eq!(end_count, END_PACKET_REPEATS as usize);
        assert_eq!(parsed.last().unwrap().duration, 800); // 100 ms @ 8 kHz
    }

    #[test]
    fn receiver_aggregates_to_single_event() {
        let mut r = DtmfReceiver::default();
        let mid = Mid::from("audio");
        let freq = Frequency::EIGHT_KHZ;

        // Playing packets (growing duration, no end bit).
        for d in [160u16, 320, 480, 640, 800] {
            r.feed(
                mid,
                MediaTime::new(1000, freq),
                TelephoneEventPayload {
                    event: 5,
                    end: false,
                    volume: 10,
                    duration: d,
                },
            );
            assert!(r.poll().is_none());
        }

        // Three end packets — only the first should produce an event.
        for _ in 0..3 {
            r.feed(
                mid,
                MediaTime::new(1000, freq),
                TelephoneEventPayload {
                    event: 5,
                    end: true,
                    volume: 10,
                    duration: 800,
                },
            );
        }

        let ev = r.poll().unwrap();
        assert_eq!(ev.event, 5);
        assert_eq!(ev.dtmf, Some(Dtmf::D5));
        assert_eq!(ev.duration.numer(), 800);
        assert!(r.poll().is_none());
    }
}
