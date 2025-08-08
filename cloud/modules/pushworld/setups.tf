#########################################################
# IAM Role for ECS Task Execution (for jobRoleArn)
#########################################################

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecs-task-execution-role"

  # Trust policy that allows ECS tasks to assume this role
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# Attachment for standard ECS task execution capabilities
resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---------------------------------------------------------------------------------
# Define the ECS Exec policy manually since it may not exist in all regions
# ---------------------------------------------------------------------------------
resource "aws_iam_policy" "ecs_exec_policy_custom" {
  name        = "ECSExecPolicy-Custom"
  description = "Provides permissions required for AWS ECS Exec functionality."

  # This JSON is the official content of the AmazonECSTaskExecuteCommandPolicy
  policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = [
          "ssmmessages:CreateControlChannel",
          "ssmmessages:CreateDataChannel",
          "ssmmessages:OpenControlChannel",
          "ssmmessages:OpenDataChannel"
        ],
        Resource = "*"
      }
    ]
  })
}

# ---------------------------------------------------------------------------------
# Attachment for enabling ECS Exec (interactive shell) using our custom policy
# ---------------------------------------------------------------------------------
resource "aws_iam_role_policy_attachment" "ecs_task_exec_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  # Use the ARN from the custom policy you just created above
  policy_arn = aws_iam_policy.ecs_exec_policy_custom.arn
}

# output "ecs_task_execution_role_arn" {
#   description = "The ARN of the ECS Task Execution Role to be used as the jobRoleArn"
#   value       = aws_iam_role.ecs_task_execution_role.arn
# }

#########################################################

resource "aws_iam_role" "ecs_instance_role" {
  name               = "ecs_instance_role"
  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
    {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
            "Service": "ec2.amazonaws.com"
        }
    }
    ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "ecs_instance_role" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "ecs_instance_role" {
  name = "ecs_instance_role"
  role = aws_iam_role.ecs_instance_role.name
}

resource "aws_iam_role" "aws_batch_service_role" {
  name = "aws_batch_service_role"

  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
    {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
        "Service": "batch.amazonaws.com"
        }
    }
    ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "aws_batch_service_role" {
  role       = aws_iam_role.aws_batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role" "AWS_EC2_spot_fleet_role" {
  name = "aws_spot_fleet_service_role"

  assume_role_policy = <<POLICY
{
    "Version": "2012-10-17",
    "Statement": [
    {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Principal": {
          "Service": "spotfleet.amazonaws.com"
        }
    }
    ]
}
POLICY
}

resource "aws_iam_role_policy_attachment" "AWS_EC2_spot_fleet_role" {
  role       = aws_iam_role.AWS_EC2_spot_fleet_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

resource "aws_security_group" "sample" {
  name = "aws_batch_compute_environment_security_group"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_default_vpc" "sample" {
  tags = {
    Name = "Default VPC"
  }
}

data "aws_subnet_ids" "all_default_subnets" {
  vpc_id = aws_default_vpc.sample.id
}