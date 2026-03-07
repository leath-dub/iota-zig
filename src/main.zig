const std = @import("std");
const heap = std.heap;
const fs = std.fs;
const log = std.log;
const mem = std.mem;

const Code = @import("Code.zig");
const Lexer = @import("Lexer.zig");
const Ast = @import("Ast.zig");
const Parser = @import("Parser.zig");
const node = @import("node.zig");
const GeneralContext = @import("GeneralContext.zig");

// Semantic passes
const ScopeBuilder = @import("ScopeBuilder.zig");

const Cli = struct {
    input_path: []const u8,
};

fn parseArgs() error{ParseFailed}!Cli {
    var cli: Cli = undefined;

    var args = std.process.args();
    _ = args.next().?;

    if (args.next()) |file_arg| {
        cli.input_path = file_arg[0..];
    } else {
        log.err("usage: iorac <file to compile>", .{});
        return error.ParseFailed;
    }

    return cli;
}

fn openErrorMsg(e: fs.File.OpenError) ?[]const u8 {
    return switch (e) {
        error.FileNotFound => "file not found",
        error.AccessDenied => "access denied",
        else => null,
    };
}

pub fn main() !void {
    var default_ctx = GeneralContext.Default.init();
    defer default_ctx.deinit();

    var ctx = default_ctx.general();

    const cli = parseArgs() catch std.process.exit(1);
    const input_file = fs.cwd().openFile(cli.input_path, .{}) catch |e| {
        const msg = openErrorMsg(e) orelse @errorName(e);
        log.err("opening file {s}: {s}\n", .{ cli.input_path, msg });
        std.process.exit(1);
    };
    defer input_file.close();

    var cst = ctx.createLifetime();
    defer cst.deinit();

    const text = try input_file.readToEndAlloc(ctx.allocator, std.math.maxInt(usize));
    var code = try Code.init(&cst, cli.input_path, try cst.allocator().dupe(u8, text));
    ctx.allocator.free(text);

    var parser = Parser.init(&ctx, Lexer.init(&ctx, &cst, &code));
    var ast = parser.parse();
    defer ast.deinit();

    if (code.errors != 0) {
        std.debug.print("{f}", .{ast});
        return error.SyntaxAnalysisFailed;
    }

    var scope_builder = ScopeBuilder.init(&ast, &code);
    defer scope_builder.deinit();
    Ast.walk(&scope_builder, &ast.root.?);

    std.debug.print("{f}", .{ast});

    if (code.errors != 0) {
        return error.SemanticAnalysisFailed;
    }
}
