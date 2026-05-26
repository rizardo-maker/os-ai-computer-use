import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:frontend_flutter/src/features/chat/application/stores/chat_store.dart';
import 'package:frontend_flutter/src/features/chat/domain/entities/chat_message.dart';
import 'package:frontend_flutter/src/features/chat/domain/entities/connection_status.dart';
import 'package:frontend_flutter/src/features/chat/domain/entities/cost_usage.dart';
import 'package:frontend_flutter/src/features/chat/domain/repositories/chat_repository.dart';

void main() {
  group('ChatStore approvals', () {
    late _FakeChatRepository repo;

    setUp(() {
      repo = _FakeChatRepository();
    });

    tearDown(() async {
      await repo.close();
    });

    test('removes approval messages for completed job', () async {
      final store = ChatStore(repo, null);

      repo.emitMessage(_approvalMessage('approval-1', 'job-1'));
      await pumpEventQueue();

      expect(store.messages.where((m) => m.kind == 'approval'), hasLength(1));

      repo.emitMessage(ChatMessage(
        id: 'control-1',
        role: 'assistant',
        ts: DateTime.now(),
        kind: 'control',
        meta: const {'removeApprovalsForJobId': 'job-1'},
      ));
      await pumpEventQueue();

      expect(store.messages.where((m) => m.kind == 'approval'), isEmpty);
    });

    test('marks stale approval response as unavailable', () async {
      final store = ChatStore(repo, null);
      repo.respondApprovalResult = false;

      repo.emitMessage(_approvalMessage('approval-1', 'job-1'));
      await pumpEventQueue();

      await store.respondApproval(
        messageId: 'approval-1',
        jobId: 'job-1',
        approvalId: 'safety-1',
        approved: true,
      );

      expect(store.messages.where((m) => m.kind == 'approval'), isEmpty);
      expect(store.messages.last.text, 'Approval is no longer active.');
      expect(store.messages.last.meta?['isError'], true);
    });

    test('records accepted approval response', () async {
      final store = ChatStore(repo, null);
      repo.respondApprovalResult = true;

      repo.emitMessage(_approvalMessage('approval-1', 'job-1'));
      await pumpEventQueue();

      await store.respondApproval(
        messageId: 'approval-1',
        jobId: 'job-1',
        approvalId: 'safety-1',
        approved: false,
      );

      expect(store.messages.where((m) => m.kind == 'approval'), isEmpty);
      expect(store.messages.last.text, 'Denied tool request.');
      expect(store.messages.last.meta, isNull);
    });
  });
}

ChatMessage _approvalMessage(String id, String jobId) {
  return ChatMessage(
    id: id,
    role: 'system',
    ts: DateTime.now(),
    kind: 'approval',
    text: 'Tool approval required',
    meta: {
      'jobId': jobId,
      'approvalId': 'safety-1',
    },
  );
}

class _FakeChatRepository implements ChatRepository {
  final _messages = StreamController<ChatMessage>.broadcast();
  final _usage = StreamController<CostUsage>.broadcast();
  final _running = StreamController<bool>.broadcast();
  final _connection = StreamController<ConnectionStatus>.broadcast();

  bool respondApprovalResult = true;

  void emitMessage(ChatMessage message) {
    _messages.add(message);
  }

  Future<void> close() async {
    await _messages.close();
    await _usage.close();
    await _running.close();
    await _connection.close();
  }

  @override
  Stream<ChatMessage> messages() => _messages.stream;

  @override
  Stream<CostUsage> usage() => _usage.stream;

  @override
  Stream<bool> running() => _running.stream;

  @override
  Stream<ConnectionStatus> connectionStatus() => _connection.stream;

  @override
  Future<String> createSession({String? provider}) async => 'session-1';

  @override
  Future<String> runTask({required String task}) async => 'job-1';

  @override
  Future<bool> respondApproval({
    required String jobId,
    required String approvalId,
    required bool approved,
  }) async {
    return respondApprovalResult;
  }

  @override
  Future<void> cancelJob(String jobId) async {}

  @override
  Future<void> cancelCurrentJob() async {}

  @override
  void setActiveChat(String chatId) {}

  @override
  Future<String> uploadFile(
    String name,
    List<int> bytes, {
    String? mime,
    void Function(int sent, int total)? onProgress,
    void Function(void Function())? onCreateCancel,
    String? previewBase64,
    String? batchId,
    int? batchSize,
    int? batchIndex,
  }) async {
    return 'file-1';
  }

  @override
  Future<List<int>> downloadFile(String id) async => <int>[];
}
